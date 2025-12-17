import streamlit as st
import time
from typing import List, Dict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --- 导入核心配置与模型接口 ---
from config.settings import settings
from core.llm import get_agent_llm, get_critic_llm 

# 兜底导入
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    pass

st.set_page_config(page_title="Auto-Debate Arena", page_icon="⚔️", layout="wide")

st.title("⚔️ Autonomous Adversarial Debate")
st.caption("观察者模式：设定初始目标，看着 AI 自己吵架直至收敛。")

# ==========================================
# 0. 辅助函数：智能模型路由 (保持不变)
# ==========================================
def get_model_instance(model_name: str, temperature: float):
    """根据名字自动路由 API (DeepSeek vs Qwen vs OpenAI)"""
    model_name_lower = model_name.lower()
    if "deepseek" in model_name_lower:
        llm = get_agent_llm(temperature=temperature)
        llm.model_name = model_name
        return llm
    elif "qwen" in model_name_lower:
        llm = get_critic_llm(temperature=temperature)
        llm.model_name = model_name
        llm.temperature = temperature 
        return llm
    else:
        return ChatOpenAI(
            model=model_name, 
            temperature=temperature, 
            api_key=settings.OPENAI_API_KEY if hasattr(settings, 'OPENAI_API_KEY') else None
        )

# ==========================================
# 1. 侧边栏配置
# ==========================================
with st.sidebar:
    st.header("⚙️ Arena Config")
    
    # 模型配置
    st.subheader("Duelists")
    builder_model = st.text_input("Builder (Proposer)", value=settings.AGENT_MODEL_NAME, help="DeepSeek-R1 推荐")
    critic_model = st.text_input("Critic (Reviewer)", value=settings.CRITIC_MODEL_NAME, help="Qwen3-Max 推荐")
    
    st.divider()
    
    # 循环控制
    max_rounds = st.slider("Max Rounds", 3, 20, 8, help="防止无限争吵")
    sleep_time = st.slider("Pacing (seconds)", 0.0, 5.0, 1.0, help="回合间隔，方便阅读")
    
    if st.button("🗑️ Reset Arena"):
        st.session_state.clear()
        st.rerun()

# ==========================================
# 2. 状态管理
# ==========================================
if "debate_history" not in st.session_state:
    st.session_state.debate_history = [] 
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# ==========================================
# 3. 核心 Prompt
# ==========================================
def get_builder_system_prompt(goal: str):
    return f"""
    You are **AI-Builder**. Your goal is to design the ULTIMATE research/technical solution.
    
    ### 🎯 CORE OBJECTIVE (FROZEN):
    "{goal}"
    
    ### Rules:
    1. If this is the first turn, propose a comprehensive plan.
    2. If you received criticism, REFINE your solution to address the flaws.
    3. DO NOT change the Core Objective.
    4. Be highly technical, precise, and logical.
    5. Please response with chinese.
    """

def get_critic_system_prompt():
    return """
    You are **AI-Critic**. Your job is to stress-test the Builder's proposal.
    
    ### Rules:
    1. Identify fatal logic gaps, feasibility issues, or security risks.
    2. Be harsh but fair. 
    3. **TERMINATION CONDITION**: If the proposal is flawless and meets all constraints effectively, output EXACTLY the word: "PASS".
    4. Otherwise, list 1-3 specific criticisms.
    5. Please response with chinese.
    """

# ==========================================
# 4. 执行逻辑 (无状态机，纯循环)
# ==========================================
def run_debate_loop(initial_goal: str):
    # 容器占位符，用于动态滚动显示
    chat_container = st.container()
    
    # 初始化历史 (仅用于本次运行的上下文构建)
    # 注意：为了让 Builder 记得之前的争论，我们需要维护一个 messages 列表
    messages_for_builder = [SystemMessage(content=get_builder_system_prompt(initial_goal))]
    
    llm_builder = get_model_instance(builder_model, 0.7)
    llm_critic = get_model_instance(critic_model, 0.5)
    
    for round_idx in range(1, max_rounds + 1):
        
        # --- A. Builder Turn ---
        with chat_container:
            with st.chat_message("assistant", avatar="👷"):
                st.write(f"**Round {round_idx}: Builder is thinking...**")
                
                # Builder 看到的是：系统指令 + 之前的对话历史
                response_a = llm_builder.invoke(messages_for_builder)
                content_a = response_a.content
                
                st.markdown(content_a)
                
                # 更新历史
                st.session_state.debate_history.append({"role": "Builder", "round": round_idx, "content": content_a})
                messages_for_builder.append(AIMessage(content=content_a))
        
        time.sleep(sleep_time)
        
        # --- B. Critic Turn ---
        with chat_container:
            with st.chat_message("assistant", avatar="🕵️"):
                st.write(f"**Round {round_idx}: Critic is reviewing...**")
                
                # Critic 只需要看到 Builder 最新的方案 (或者你可以选择给它看全部，但通常只看最新的方案更容易聚焦)
                # 这里我们构建一个新的 prompt 给 Critic
                critic_input = [
                    SystemMessage(content=get_critic_system_prompt()),
                    HumanMessage(content=f"Here is the Builder's latest proposal (Round {round_idx}):\n\n{content_a}\n\nEvaluate it.")
                ]
                
                response_b = llm_critic.invoke(critic_input)
                content_b = response_b.content
                
                # 检查是否通过
                is_pass = "PASS" in content_b or "pass" in content_b if len(content_b) < 50 else False
                
                if is_pass:
                    st.success("✅ **PASS**: Critic has approved the proposal!")
                else:
                    st.markdown(content_b)
                
                # 更新历史
                st.session_state.debate_history.append({"role": "Critic", "round": round_idx, "content": content_b})
                
                # 关键：把 Critic 的意见反馈给 Builder
                messages_for_builder.append(HumanMessage(content=f"Critic's Feedback: {content_b}\n\nPlease refine the solution."))

        if is_pass:
            st.balloons()
            return # 结束循环

        time.sleep(sleep_time)
        st.divider() # 视觉分割线

    st.warning(f"⚠️ Reached max rounds ({max_rounds}) without full consensus.")

# ==========================================
# 5. 主界面
# ==========================================
st.info("💡 **Tip**: Enter a vague idea (e.g., 'A flying car') and watch them turn it into a concrete spec.")

user_goal = st.text_area("Initial Research Goal / Hypothesis", height=100, placeholder="e.g. I want to use Reinforcement Learning to optimize SQL queries in real-time...")

col1, col2 = st.columns([1, 5])
with col1:
    start_btn = st.button("🚀 Ignite Debate", type="primary", disabled=st.session_state.is_running)

# 如果有历史记录，先展示出来 (保证刷新后不消失)
if st.session_state.debate_history and not start_btn:
    for item in st.session_state.debate_history:
        avatar = "👷" if item["role"] == "Builder" else "🕵️"
        with st.chat_message("assistant", avatar=avatar):
            st.caption(f"Round {item['round']} - {item['role']}")
            if item["content"] == "PASS":
                 st.success("✅ PASS")
            else:
                 st.markdown(item["content"])

# 点击开始后
if start_btn and user_goal:
    # 清空旧历史
    st.session_state.debate_history = []
    st.session_state.is_running = True
    
    # 运行循环
    try:
        run_debate_loop(user_goal)
    except Exception as e:
        st.error(f"❌ Error detected: {e}")
    finally:
        st.session_state.is_running = False