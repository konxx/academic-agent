import streamlit as st
import time
import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --- 核心模块导入 ---
from config.settings import settings
from core.llm import get_agent_llm, get_critic_llm

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    pass

st.set_page_config(page_title="Idea Debate Arena", page_icon="⚔️", layout="wide")

st.title("⚔️ 论文想法辩论")
st.caption("输入研究想法，通过 AI 对抗性辩论打磨出更严谨、更有创新性的论文方案")

# ==========================================
# 数据结构定义
# ==========================================
@dataclass
class CriticScore:
    novelty: int = 0
    soundness: int = 0
    significance: int = 0
    experiments: int = 0
    overall: int = 0
    verdict: str = "REVISE"
    key_issues: List[str] = None
    focus_next: str = ""
    
    def __post_init__(self):
        if self.key_issues is None:
            self.key_issues = []

@dataclass  
class DebateRound:
    round_num: int
    builder_response: str
    critic_response: str
    scores: CriticScore
    timestamp: float = 0.0

BUILDER_SYSTEM_PROMPT = """你是 **论文构建者 (Builder)**，职责是将用户的研究想法发展成一个严谨、可行的学术论文方案。

## 研究想法 (核心不可偏离)
{goal}

## 输出格式要求 (必须严格遵守)
请按以下学术论文结构输出你的方案：

### 🔬 研究问题 (Research Question)
- 明确要解决的核心科学问题
- 现有方法的不足之处 (Research Gap)

### 💡 主要创新点 (Contributions)
- [列出 2-3 个明确的学术贡献]
- 说明与现有工作的差异

### � 方法论 (Methodology)
- 提出的方法/模型/框架
- 关键技术细节
- 理论依据或原理

### 🧪 实验设计 (Experiments)
- 数据集选择及理由
- 基线对比方法 (Baselines)
- 评估指标 (Metrics)
- 消融实验计划 (Ablation Study)

### ⚠️ 潜在局限性 (Limitations)
- 主动识别可能的弱点
- 应对或缓解策略

## 规则
1. **首轮**：提出完整的论文研究方案
2. **后续轮**：针对 Critic 的学术批评进行定向改进
3. 保持学术严谨性，引用相关工作时要具体
4. 用中文回复"""

CRITIC_SYSTEM_PROMPT = """你是 **学术审稿人 (Critic)**，职责是以顶会/顶刊审稿人的标准严格评估论文方案。

## 评估维度 (对标顶会审稿标准)
1. **novelty (创新性)**: 研究问题是否新颖？方法是否有原创性？是否只是简单的组合？
2. **soundness (理论严谨性)**: 方法论是否有理论支撑？逻辑是否完整？有无明显漏洞？
3. **significance (重要性)**: 研究问题是否重要？能否推动领域发展？
4. **experiments (实验设计)**: 实验是否充分？baselines选择是否合理？评估指标是否恰当？

## 输出格式 (必须输出有效JSON)
```json
{
  "scores": {
    "novelty": <1-10分>,
    "soundness": <1-10分>,
    "significance": <1-10分>,
    "experiments": <1-10分>
  },
  "overall": <1-10分，综合评分>,
  "verdict": "<PASS 或 REVISE>",
  "key_issues": ["具体学术问题1", "具体学术问题2"],
  "focus_next": "下一轮应重点改进的学术问题"
}
```

## 评审标准
- **Strong Accept (9-10)**: 重大创新，方法严谨，实验充分
- **Accept (7-8)**: 有创新点，方法合理，实验基本完整
- **Borderline (5-6)**: 创新有限，存在一些问题但可修改
- **Reject (1-4)**: 创新不足或存在严重问题

## 通过标准
- 当 overall >= 8 且无 key_issues 时，verdict = "PASS" (相当于 Accept)
- 否则 verdict = "REVISE"

## 规则
1. 像真实审稿人一样提出建设性批评
2. 指出与现有工作的可能重叠 (如有)
3. 质疑实验设计的合理性
4. 承认 Builder 的改进进步
5. 用中文填写 key_issues 和 focus_next"""

# ==========================================
# 辅助函数
# ==========================================
def get_model_instance(model_name: str, temperature: float):
    """智能模型路由"""
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

def parse_critic_response(response: str) -> CriticScore:
    """解析 Critic 的 JSON 响应"""
    try:
        # 尝试提取 JSON 块
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试直接解析
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found")
        
        data = json.loads(json_str)
        scores = data.get("scores", {})
        
        return CriticScore(
            novelty=scores.get("novelty", 5),
            soundness=scores.get("soundness", 5),
            significance=scores.get("significance", 5),
            experiments=scores.get("experiments", 5),
            overall=data.get("overall", 5),
            verdict=data.get("verdict", "REVISE"),
            key_issues=data.get("key_issues", []),
            focus_next=data.get("focus_next", "")
        )
    except Exception as e:
        # Fallback: 如果解析失败，返回默认评分
        return CriticScore(
            overall=5,
            verdict="REVISE",
            key_issues=["JSON解析失败，请检查Critic输出格式"],
            focus_next="继续改进方案"
        )

def create_score_chart(rounds: List[DebateRound]) -> go.Figure:
    """创建评分趋势图"""
    if not rounds:
        return None
    
    round_nums = [r.round_num for r in rounds]
    overall_scores = [r.scores.overall for r in rounds]
    novelty = [r.scores.novelty for r in rounds]
    soundness = [r.scores.soundness for r in rounds]
    significance = [r.scores.significance for r in rounds]
    experiments = [r.scores.experiments for r in rounds]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=round_nums, y=overall_scores,
        mode='lines+markers',
        name='综合评分',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=round_nums, y=novelty,
        mode='lines+markers',
        name='创新性',
        line=dict(color='#4ECDC4', width=2, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=round_nums, y=soundness,
        mode='lines+markers', 
        name='严谨性',
        line=dict(color='#45B7D1', width=2, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=round_nums, y=significance,
        mode='lines+markers',
        name='重要性',
        line=dict(color='#96CEB4', width=2, dash='dot')
    ))
    
    fig.add_trace(go.Scatter(
        x=round_nums, y=experiments,
        mode='lines+markers',
        name='实验设计',
        line=dict(color='#FFEAA7', width=2, dash='dot')
    ))
    
    # 添加通过线
    fig.add_hline(y=8, line_dash="dash", line_color="green", 
                  annotation_text="通过线 (8分)")
    
    fig.update_layout(
        title="📈 评分趋势",
        xaxis_title="轮次",
        yaxis_title="评分",
        yaxis=dict(range=[0, 10.5]),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3)
    )
    
    return fig

def generate_summary(rounds: List[DebateRound], goal: str) -> str:
    """生成辩论总结"""
    if not rounds:
        return ""
    
    final_round = rounds[-1]
    total_rounds = len(rounds)
    final_score = final_round.scores.overall
    status = "✅ 方案通过" if final_round.scores.verdict == "PASS" else "⏸️ 达到轮次上限"
    
    # 计算进步幅度
    if len(rounds) > 1:
        improvement = final_score - rounds[0].scores.overall
        improvement_text = f"+{improvement}" if improvement > 0 else str(improvement)
    else:
        improvement_text = "N/A"
    
    summary = f"""
## 📋 辩论总结

| 指标 | 值 |
|-----|-----|
| 核心目标 | {goal[:50]}... |
| 总轮次 | {total_rounds} |
| 最终评分 | {final_score}/10 |
| 评分变化 | {improvement_text} |
| 最终状态 | {status} |

### 最终方案摘要
{final_round.builder_response[:500]}...
"""
    return summary

def export_to_markdown(rounds: List[DebateRound], goal: str) -> str:
    """导出辩论记录为 Markdown"""
    md = f"# 辩论记录\n\n## 核心目标\n{goal}\n\n---\n\n"
    
    for r in rounds:
        md += f"## 第 {r.round_num} 轮\n\n"
        md += f"### 👷 Builder 方案\n{r.builder_response}\n\n"
        md += f"### 🕵️ Critic 评审\n"
        md += f"- 综合评分: {r.scores.overall}/10\n"
        md += f"- 重要性: {r.scores.novelty}/10\n"
        md += f"- 严谨性: {r.scores.soundness}/10\n"
        md += f"- 创新性: {r.scores.significance}/10\n"
        md += f"- 实验设计: {r.scores.experiments}/10\n"
        if r.scores.key_issues:
            md += f"- 关键问题: {', '.join(r.scores.key_issues)}\n"
        md += f"\n---\n\n"
    
    return md

# ==========================================
# 侧边栏配置
# ==========================================
with st.sidebar:
    st.header("⚙️ 辩论配置")
    
    st.subheader("🤖 模型选择")
    builder_model = st.text_input(
        "Builder (方案设计)", 
        value=settings.AGENT_MODEL_NAME,
        help="推荐使用 DeepSeek-R1"
    )
    critic_model = st.text_input(
        "Critic (方案审查)", 
        value=settings.CRITIC_MODEL_NAME,
        help="推荐使用 Qwen-Max"
    )
    
    st.divider()
    
    st.subheader("🎚️ 参数调节")
    max_rounds = st.slider("最大轮次", 3, 15, 8, help="防止无限辩论")
    builder_temp = st.slider("Builder 创造性", 0.0, 1.0, 0.7, 0.1)
    critic_temp = st.slider("Critic 严格度", 0.0, 1.0, 0.3, 0.1, help="越低越严格")
    
    st.divider()
    
    if st.button("🗑️ 清空辩论", type="secondary"):
        st.session_state.clear()
        st.rerun()

# ==========================================
# Session State 初始化
# ==========================================
if "debate_rounds" not in st.session_state:
    st.session_state.debate_rounds: List[DebateRound] = []
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "current_goal" not in st.session_state:
    st.session_state.current_goal = ""

# ==========================================
# 主界面
# ==========================================
st.subheader("1. 设定研究目标")
user_goal = st.text_area(
    "描述你的研究目标或技术想法",
    height=100,
    placeholder="例如：设计一个基于强化学习的SQL查询实时优化系统，要求能处理高并发场景..."
)

col_start, col_status = st.columns([1, 3])
with col_start:
    start_btn = st.button(
        "🚀 开始辩论", 
        type="primary", 
        disabled=st.session_state.is_running or not user_goal
    )
with col_status:
    if st.session_state.is_running:
        st.info("🔄 辩论进行中...")
    elif st.session_state.debate_rounds:
        final = st.session_state.debate_rounds[-1]
        if final.scores.verdict == "PASS":
            st.success(f"✅ 辩论完成！最终评分: {final.scores.overall}/10")
        else:
            st.warning(f"⏸️ 达到最大轮次，当前评分: {final.scores.overall}/10")

st.divider()

# ==========================================
# 辩论执行逻辑
# ==========================================
if start_btn and user_goal:
    st.session_state.debate_rounds = []
    st.session_state.is_running = True
    st.session_state.current_goal = user_goal
    
    # 初始化模型
    llm_builder = get_model_instance(builder_model, builder_temp)
    llm_critic = get_model_instance(critic_model, critic_temp)
    
    # 初始化对话历史 - 双方都维护完整历史
    builder_history = [SystemMessage(content=BUILDER_SYSTEM_PROMPT.format(goal=user_goal))]
    critic_history = [SystemMessage(content=CRITIC_SYSTEM_PROMPT)]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 创建展示容器
    col_chat, col_stats = st.columns([2, 1])
    
    with col_stats:
        chart_placeholder = st.empty()
        stats_placeholder = st.empty()
    
    with col_chat:
        chat_container = st.container()
    
    try:
        for round_idx in range(1, max_rounds + 1):
            progress_bar.progress(round_idx / max_rounds)
            
            # === Builder 回合 ===
            status_text.text(f"🔄 第 {round_idx} 轮: Builder 正在设计方案...")
            
            with chat_container:
                with st.chat_message("assistant", avatar="👷"):
                    st.caption(f"**第 {round_idx} 轮 - Builder**")
                    builder_placeholder = st.empty()
                    builder_placeholder.text("思考中...")
            
            builder_response = llm_builder.invoke(builder_history)
            builder_content = builder_response.content
            builder_history.append(AIMessage(content=builder_content))
            
            with chat_container:
                with st.chat_message("assistant", avatar="👷"):
                    st.caption(f"**第 {round_idx} 轮 - Builder**")
                    st.markdown(builder_content)
            
            time.sleep(0.5)
            
            # === Critic 回合 ===
            status_text.text(f"🔄 第 {round_idx} 轮: Critic 正在审查...")
            
            # Critic 看到完整辩论历史
            critic_history.append(HumanMessage(
                content=f"## 第 {round_idx} 轮 Builder 方案\n\n{builder_content}\n\n请按JSON格式评估此方案。"
            ))
            
            with chat_container:
                with st.chat_message("assistant", avatar="🕵️"):
                    st.caption(f"**第 {round_idx} 轮 - Critic**")
                    critic_placeholder = st.empty()
                    critic_placeholder.text("审查中...")
            
            critic_response = llm_critic.invoke(critic_history)
            critic_content = critic_response.content
            critic_history.append(AIMessage(content=critic_content))
            
            # 解析评分
            scores = parse_critic_response(critic_content)
            
            # 记录本轮结果
            debate_round = DebateRound(
                round_num=round_idx,
                builder_response=builder_content,
                critic_response=critic_content,
                scores=scores,
                timestamp=time.time()
            )
            st.session_state.debate_rounds.append(debate_round)
            
            # 展示 Critic 结果
            with chat_container:
                with st.chat_message("assistant", avatar="🕵️"):
                    st.caption(f"**第 {round_idx} 轮 - Critic**")
                    
                    # 评分卡片
                    score_cols = st.columns(5)
                    score_cols[0].metric("综合", f"{scores.overall}/10")
                    score_cols[1].metric("创新性", f"{scores.novelty}/10")
                    score_cols[2].metric("严谨性", f"{scores.soundness}/10")
                    score_cols[3].metric("重要性", f"{scores.significance}/10")
                    score_cols[4].metric("实验", f"{scores.experiments}/10")
                    
                    if scores.verdict == "PASS":
                        st.success("✅ **PASS** - 方案已通过审核！")
                    else:
                        if scores.key_issues:
                            st.warning("**待改进问题:**")
                            for issue in scores.key_issues:
                                st.markdown(f"- {issue}")
                        if scores.focus_next:
                            st.info(f"**下轮焦点:** {scores.focus_next}")
            
            # 更新图表
            with col_stats:
                fig = create_score_chart(st.session_state.debate_rounds)
                if fig:
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # 更新统计
                stats_placeholder.markdown(f"""
                ### 📊 当前状态
                - **轮次**: {round_idx}/{max_rounds}
                - **评分**: {scores.overall}/10
                - **状态**: {"🟢 通过" if scores.verdict == "PASS" else "🟡 修订中"}
                """)
            
            # 检查是否通过
            if scores.verdict == "PASS" and scores.overall >= 8:
                status_text.text("🎉 辩论收敛，方案通过！")
                st.balloons()
                break
            
            # 将 Critic 反馈给 Builder
            feedback = f"""
## Critic 评审结果 (第 {round_idx} 轮)
- 综合评分: {scores.overall}/10
- 关键问题: {', '.join(scores.key_issues) if scores.key_issues else '无'}
- 改进焦点: {scores.focus_next}

请针对以上问题改进你的方案。
"""
            builder_history.append(HumanMessage(content=feedback))
            
            time.sleep(0.5)
        
        else:
            status_text.text(f"⚠️ 达到最大轮次 ({max_rounds})，辩论结束")
        
        progress_bar.empty()
        
    except Exception as e:
        st.error(f"❌ 辩论过程出错: {str(e)}")
    finally:
        st.session_state.is_running = False

# ==========================================
# 展示历史记录 (非运行状态)
# ==========================================
if st.session_state.debate_rounds and not st.session_state.is_running:
    st.subheader("2. 辩论进程")
    
    col_history, col_analysis = st.columns([2, 1])
    
    with col_analysis:
        # 评分趋势图
        fig = create_score_chart(st.session_state.debate_rounds)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # 辩论总结
        summary = generate_summary(
            st.session_state.debate_rounds,
            st.session_state.current_goal
        )
        st.markdown(summary)
        
        # 导出按钮
        md_content = export_to_markdown(
            st.session_state.debate_rounds,
            st.session_state.current_goal
        )
        st.download_button(
            "📥 导出 Markdown",
            data=md_content,
            file_name="debate_record.md",
            mime="text/markdown"
        )
    
    with col_history:
        # 角色筛选
        filter_role = st.radio(
            "筛选角色",
            ["全部", "只看 Builder", "只看 Critic"],
            horizontal=True
        )
        
        for r in st.session_state.debate_rounds:
            with st.expander(f"📍 第 {r.round_num} 轮 (评分: {r.scores.overall}/10)", expanded=(r.round_num == len(st.session_state.debate_rounds))):
                if filter_role in ["全部", "只看 Builder"]:
                    st.markdown("#### 👷 Builder")
                    st.markdown(r.builder_response)
                
                if filter_role in ["全部", "只看 Critic"]:
                    st.markdown("#### 🕵️ Critic")
                    score_cols = st.columns(5)
                    score_cols[0].metric("综合", f"{r.scores.overall}/10")
                    score_cols[1].metric("创新性", f"{r.scores.novelty}/10")
                    score_cols[2].metric("严谨性", f"{r.scores.soundness}/10")
                    score_cols[3].metric("重要性", f"{r.scores.significance}/10")
                    score_cols[4].metric("实验", f"{r.scores.experiments}/10")
                    
                    if r.scores.key_issues:
                        st.warning("**问题:** " + ", ".join(r.scores.key_issues))
                    if r.scores.focus_next:
                        st.info(f"**下轮焦点:** {r.scores.focus_next}")