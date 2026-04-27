import os
import asyncio
import time as _time
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# ==================== 1. 加载 lightrag-server 的 .env ====================
# 假设 .env 文件和脚本在同一目录；如果在别处，改成 load_dotenv("/path/to/.env")
load_dotenv("D:\develop\LightRAG\.env")

# ==================== 2. 根据 .env 自动构造 Embedding 函数 ====================
def build_embedding_func_from_env():
    binding = os.getenv("EMBEDDING_BINDING", "openai").lower()
    model   = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    dim     = int(os.getenv("EMBEDDING_DIM", "1536"))
    host    = os.getenv("EMBEDDING_BINDING_HOST", "")
    api_key = os.getenv("EMBEDDING_BINDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    max_token_size = os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "8192")
    print(f"Embedding config - binding: {binding}, model: {model}, dim: {dim}, host: {host}, max_token_size: {max_token_size}")
    if binding == "openai":
        from lightrag.llm.openai import openai_embed
        from lightrag.utils import wrap_embedding_func_with_attrs
        print(f"dim: {dim}")
        @wrap_embedding_func_with_attrs(embedding_dim=dim, max_token_size=max_token_size, model_name=model)
        async def embedding_func(texts: list[str]):
            return await openai_embed.func(texts, model=model, api_key=api_key, base_url=host)
    else:
        raise ValueError(f"不支持的 EMBEDDING_BINDING: {binding}")

    return embedding_func


# ==================== 3. 根据 .env 自动构造 LLM 函数 ====================
def build_llm_func_from_env():
    binding = os.getenv("LLM_BINDING", "openai").lower()
    model   = os.getenv("LLM_MODEL", "gpt-4o-mini")
    host    = os.getenv("LLM_BINDING_HOST", "")
    api_key = os.getenv("LLM_BINDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))

    if binding == "openai":
        from lightrag.llm.openai import openai_complete_if_cache
        async def _llm(prompt, system_prompt=None, history_messages=[], **kwargs):
            return await openai_complete_if_cache(
                model, prompt, system_prompt=system_prompt,
                history_messages=history_messages, api_key=api_key, base_url=host,
                **kwargs
            )
    else:
        raise ValueError(f"不支持的 LLM_BINDING: {binding}")

    return _llm


# ==================== 4. 初始化 LightRAG（复用 server 的存储） ====================
async def init_rag_from_env():
    """
    working_dir 必须和 lightrag-server 的一致。
    如果 server 通过 --working-dir 传入，这里也要用相同路径。
    """
    working_dir = os.getenv("WORKING_DIR", "./rag_storage")

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=build_llm_func_from_env(),
        embedding_func=build_embedding_func_from_env(),
    )

    # 这两个初始化调用缺一不可 [^137^][^139^]
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


# ==================== 5. 语义检索 FaultSymptom + 图遍历 CheckStep ====================
async def build_check_flow(rag, user_query: str, top_k_symptoms: int = 3):
    """
    根据用户输入语义查找最相似的 FaultSymptom，
    遍历 FaultCategory 收集 RootCause，按 Category 聚合排序后由 LLM 判定最可能的类别，
    最后组装该类别的 CheckStep 检查流程返回给用户。
    """
    graph = rag.chunk_entity_relation_graph
    results = []

    # ---------- 5.1 & 5.2 语义检索 FaultSymptom ----------
    raw_results = await rag.entities_vdb.query(user_query, top_k=15)
    # print(f"语义检索原始结果: {raw_results}")

    symptoms = []
    for r in raw_results:
        entity_name = r.get("entity_name") or r.get("id")
        node = await graph.get_node(entity_name)
        # print(f"检索结果实体: {entity_name}\n 节点数据: {node}")
        if node and node.get("entity_type") == "faultsymptom":
            symptoms.append((r.get("distance"), node))
    print(f"语义检索到的 FaultSymptom : {symptoms}")

    # ---------- 5.3 遍历 FaultCategory 并收集 RootCause（按 Category 分组） ----------
    symptoms_with_categories = []
    for sim_score, symptom in symptoms:
        symptom_id = (
            symptom.get("entity_id")
            or symptom.get("id")
            or symptom.get("entity_name")
        )
        edges = await graph.get_node_edges(symptom_id) or []
        print(f"edges: {edges}")
        faultcategories = []

        for src_id, tgt_id in edges:
            neighbor_id = tgt_id if src_id == symptom_id else src_id
            neighbor = await graph.get_node(neighbor_id)
            print(f"邻居节点: {neighbor_id}\n 数据: {neighbor}")

            if neighbor and neighbor.get("entity_type") == "faultcategory":
                edge_data = await graph.get_edge(symptom_id, neighbor_id) or {}

                # 读取 faultcategory 节点的时间信息
                category_updated_at = neighbor.get("updated_at", 0) or 0
                category_created_at = neighbor.get("created_at", 0) or 0

                # 遍历该 Category 的边，找 RootCause
                cat_edges = await graph.get_node_edges(neighbor_id) or []
                rootcauses = []
                for cat_src, cat_tgt in cat_edges:
                    root_id = cat_tgt if cat_src == neighbor_id else cat_src
                    root_node = await graph.get_node(root_id)
                    if root_node and root_node.get("entity_type") == "rootcause":
                        visit_count = (
                            root_node.get("visit_count")
                            or root_node.get("access_count")
                            or 0
                        )
                        last_time = (
                            root_node.get("last_visit_time")
                            or root_node.get("last_time")
                            or root_node.get("timestamp", "")
                        )
                        # 也读取 rootcause 节点的 updated_at
                        rootcause_updated_at = root_node.get("updated_at", 0) or 0
                        rootcauses.append({
                            "rootcause_id": root_id,
                            "rootcause_name": root_node.get("entity_name", root_id),
                            "description": root_node.get("description", ""),
                            "visit_count": int(visit_count) if visit_count else 0,
                            "last_time": str(last_time) if last_time else "",
                            "updated_at": rootcause_updated_at,
                        })

                faultcategories.append({
                    "category_id": neighbor_id,
                    "category_name": neighbor.get("entity_name", neighbor_id),
                    "description": neighbor.get("description", ""),
                    "relation_weight": edge_data.get("weight", 0),
                    "relation_keywords": edge_data.get("keywords", ""),
                    "source_chunk": neighbor.get("source_id", ""),
                    "updated_at": category_updated_at,
                    "created_at": category_created_at,
                    "rootcauses": rootcauses,
                })

        symptoms_with_categories.append({
            "fault_symptom": symptom,
            "similarity_score": sim_score,
            "faultcategories": faultcategories,
        })
    print(f"symptoms_with_categories: {symptoms_with_categories}")

    # ---------- 5.4 按 FaultCategory 聚合排序（含时间衰减） ----------

    def parse_time(time_str):
        if not time_str or time_str in ("None", "null", ""):
            return 0
        try:
            if "T" in time_str:
                return datetime.fromisoformat(time_str.replace("Z", "+00:00")).timestamp()
            else:
                return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            return 0

    # 时间衰减函数：half_life_days 为半衰期天数，updated_at 为 unix timestamp
    # 返回 0~1 的衰减因子，越新越接近 1
    TIME_DECAY_HALF_LIFE_DAYS = 30  # 半衰期 30 天，可根据业务调整

    def time_decay_score(updated_at: int, half_life_days: float = TIME_DECAY_HALF_LIFE_DAYS) -> float:
        if not updated_at or updated_at <= 0:
            return 0.0
        age_seconds = _time.time() - updated_at
        if age_seconds < 0:
            return 1.0
        return 0.5 ** (age_seconds / (half_life_days * 86400))

    category_candidates = []
    for item in symptoms_with_categories:
        symptom = item["fault_symptom"]
        sim_score = item["similarity_score"]
        symptom_id = (
            symptom.get("entity_id")
            or symptom.get("id")
            or symptom.get("entity_name")
        )
        for cat in item["faultcategories"]:
            rootcauses = cat.get("rootcauses", [])
            total_visits = sum(rc["visit_count"] for rc in rootcauses)

            # 综合 FaultCategory 节点和其下 RootCause 节点的 updated_at，
            # 取最新一个作为该候选类别的最新更新时间
            all_updated_ats = [cat.get("updated_at", 0) or 0]
            all_updated_ats.extend(rc.get("updated_at", 0) or 0 for rc in rootcauses)
            latest_updated_at = max(all_updated_ats)

            # 综合评分 = 语义相似度权重 + 访问频次权重 + 时间衰减权重
            # 语义相似度: distance 越大越好，取倒数
            sim_component = sim_score if sim_score else 0.0
            # 访问频次: log 平滑，避免大数值主导
            import math
            visit_component = math.log1p(total_visits)
            # 时间衰减: 越新越高
            time_component = time_decay_score(latest_updated_at)

            # 综合评分（权重可调）
            composite_score = (
                0.6 * sim_component
                + 0.2 * visit_component
                + 0.2 * time_component
            )

            category_candidates.append({
                "category": cat,
                "symptom": symptom,
                "symptom_id": symptom_id,
                "similarity_score": sim_score,
                "total_visits": total_visits,
                "latest_updated_at": latest_updated_at,
                "time_decay_score": time_component,
                "composite_score": composite_score,
                "rootcauses": rootcauses,
            })

    if not category_candidates:
        print("未找到任何 FaultCategory，返回空结果")
        return results
    print(f"categories: {category_candidates}")
    # 按综合评分降序排序
    category_candidates.sort(
        key=lambda c: c["composite_score"],
        reverse=True
    )
    topk_cats = category_candidates[:top_k_symptoms]
    print(f"排序后 Top-{len(topk_cats)} FaultCategory: {[c['category']['category_name'] for c in topk_cats]}")

    # ---------- 5.5 将 FaultCategory 和用户输入交给 LLM，判断最可能的类别 ----------
    best_category = topk_cats[0] if topk_cats else None

    if len(topk_cats) > 1:
        candidates_text = "\n".join([
            f"{i+1}. {c['category']['category_name']}（描述: {c['category']['description']}，"
            f"关联根因数: {len(c['rootcauses'])}，总访问: {c['total_visits']}，"
            f"综合评分: {c['composite_score']:.3f}，时间衰减: {c['time_decay_score']:.3f}）"
            for i, c in enumerate(topk_cats)
        ])

        prompt = (
            f"用户描述的故障现象如下：\n{user_query}\n\n"
            f"通过知识图谱检索到以下候选故障类别（已按历史访问频次和最近访问时间排序）：\n"
            f"{candidates_text}\n\n"
            f"请根据故障现象的专业特征，判断最可能的故障类别是哪一个。"
            f"只需返回编号数字（1-{len(topk_cats)}），不要任何解释。"
        )

        try:
            llm_response = await rag.llm_model_func(prompt)
            print(f"LLM 判断结果: {llm_response}")

            import re
            match = re.search(r'\d+', str(llm_response))
            if match:
                idx = int(match.group()) - 1
                if 0 <= idx < len(topk_cats):
                    best_category = topk_cats[idx]
        except Exception as e:
            print(f"LLM 调用失败，回退到排序第一的类别: {e}")

    # ---------- 5.6 从 RootCause 的邻居中收集 CheckStep ----------
    target_cat = best_category["category"]
    target_symptom = best_category["symptom"]
    target_symptom_id = best_category["symptom_id"]

    check_steps = []
    seen_step_ids = set()
    print(f"选定的 FaultCategory: {best_category}")
    for rootcause in best_category.get("rootcauses", []):
        rc_id = rootcause["rootcause_id"]
        edges = await graph.get_node_edges(rc_id) or []
        for src_id, tgt_id in edges:
            neighbor_id = tgt_id if src_id == rc_id else src_id
            if neighbor_id in seen_step_ids:
                continue
            neighbor = await graph.get_node(neighbor_id)
            if neighbor and neighbor.get("entity_type") == "checkstep":
                seen_step_ids.add(neighbor_id)
                edge_data = await graph.get_edge(rc_id, neighbor_id) or {}

                # 查找该 CheckStep 关联的 CodeAction 和 Conclusion
                code_actions = []
                conclusions = []
                step_edges = await graph.get_node_edges(neighbor_id) or []
                for s_src, s_tgt in step_edges:
                    code_neighbor_id = s_tgt if s_src == neighbor_id else s_src
                    code_node = await graph.get_node(code_neighbor_id)
                    if not code_node:
                        continue
                    entity_type = code_node.get("entity_type")
                    if entity_type == "codeaction":
                        code_actions.append(code_node.get("description", ""))
                    elif entity_type == "conclusion":
                        conclusions.append(code_node.get("description", ""))

                # 拼接命令和结论到描述中
                desc = neighbor.get("description", "")
                parts = [desc] if desc else []
                parts.append("执行的命令：")
                if code_actions:
                    parts.append("\n".join(f"{cmd}" for cmd in code_actions if cmd))
                parts.append("检查步骤的结论：")
                if conclusions:
                    parts.append("\n".join(con for con in conclusions if con))
                desc = "\n".join(parts)

                check_steps.append({
                    "step_id": neighbor_id,
                    "step_name": neighbor.get("entity_name", neighbor_id),
                    "description": desc,
                    "relation_weight": edge_data.get("weight", 0),
                    "relation_keywords": edge_data.get("keywords", ""),
                    "source_chunk": neighbor.get("source_id", ""),
                    "rootcause_id": rc_id,
                    "rootcause_name": rootcause["rootcause_name"],
                })
    print(f"收集到的 CheckStep : {check_steps}")
    # 按关系权重排序，形成检查流程
    check_steps.sort(key=lambda x: x["relation_weight"], reverse=True)

    results.append({
        "fault_symptom": {
            "id": target_symptom_id,
            "name": target_symptom.get("entity_name", target_symptom_id),
            "description": target_symptom.get("description", ""),
            "similarity_score": best_category["similarity_score"],
        },
        "fault_category": {
            "id": target_cat["category_id"],
            "name": target_cat["category_name"],
            "description": target_cat["description"],
            "relation_weight": target_cat["relation_weight"],
            "relation_keywords": target_cat["relation_keywords"],
        },
        "root_causes": target_cat.get("rootcauses", []),
        "check_flow": check_steps,
    })

    return results


# ==================== 6. 主函数 ====================
async def main():
    rag = await init_rag_from_env()

    # 用户描述的症状（可以是自然语言，不需要精确匹配节点名）
    user_input = "Pod 处于 ContainerCreating 状态"
    flows = await build_check_flow(rag, user_input, top_k_symptoms=20)

    for flow in flows:
        fs = flow["fault_symptom"]
        print(f"\n=== 故障症状: {fs['name']} (相似度: {fs['similarity_score']}) ===")
        print(f"描述: {fs['description']}")
        print("检查流程:")
        for i, step in enumerate(flow["check_flow"], 1):
            print(f"  {i}. {step['step_name']} (关系权重: {step['relation_weight']})")
            print(f"     描述: {step['description']}")
            if step["relation_keywords"]:
                print(f"     关系关键词: {step['relation_keywords']}")

    # 清理
    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())