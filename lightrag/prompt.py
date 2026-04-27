from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
            - For a **FaultSymptom** (e.g., a pod crash, node not ready, image pull error): Use the format `[resource_type]_[failure_description]_[event_description]`. Example: `Pod_CrashLoopBackOff_[xxx]`, `Node_NotReady_[xxx]`, `Pod_ImagePullBackOff_[xxx]`, `Service_NoEndpoints_[xxx]`.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    *   **Allowed Relationship Schema (Strict Whitelist):**  
        You MUST ONLY extract relationships that exactly match one of the following six binary schemas. Any relationship not conforming to these `(source_type → target_type)` constraints is considered invalid and MUST be discarded. Do NOT invent new relationship types, and do NOT output `Other`.

        | Relationship Type | Source Entity Type | Target Entity Type | Semantic Meaning |
        |---|---|---|---|
        | `BELONGS_TO` | `FaultSymptom` | `FaultCategory` | 现象属于某故障类别导致 |
        | `HAS_ROOT_CAUSE` | `FaultCategory` | `RootCause` | 类别包含可能的根因 |
        | `REQUIRES_STEP` | `RootCause` | `CheckStep` | 排查该原因需要的步骤 |
        | `USES_CODE` | `CheckStep` | `CodeAction` | 步骤对应的代码或命令 |
        | `LEADS_TO` | `CheckStep` | `Conclusion` | 步骤得出的结论 |
        | `NEXT_STEP` | `CheckStep` | `CheckStep` | 步骤间的前后执行顺序 |
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
        *   **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `relationship_keywords`: Categorize the relationship using one of the following types: `{relationship_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Other`. One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection.
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}
"""

PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract entities and relationships from the input text in Data to be Processed below.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.
4.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
"""

PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
    *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
    *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
3.  **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""

PROMPTS["entity_extraction_examples"] = [
    """<Entity_types>
["FaultSymptom","FaultCategory","RootCause","CheckStep","CodeAction","Conclusion"]

<Input Text>
```
故障现象：Pod 处于 ContainerCreating 状态, Event reason 提示挂卷失败报错all targetPortals not available。
排查方法：1. 获取故障 Pod 信息和事件\n\n**目标：** 获取故障 Pod 的信息以及事件。\n\n**操作：** 获取 Pod 的 JSON 数据，同时获取与该 Pod 相关的事件。\n\n**需要执行的命令：**\n```bash\nkubectl get pod <pod_name> -n <namespace> -o json\nkubectl get events --field-selector=involvedObject.name=<pod_name> -n <namespace> --sort-by='.lastTimestamp' -o json\n```\n\n---\n\n2. 获取Pod对应的数据节点信息\n\n**目标：** 获取故障Pod所在的数据节点信息和node_ip。\n\n**操作：** 从上一个步骤中的Pod json得到nodename。并执行`kubectl get node <nodename> -ojson`，读取其中的node_ip\n\n**需要执行的命令：**无\n\n3. 在数据节点上获取磁阵的ip地址\n\n**目标：** 获取节点上磁阵的ip session。\n\n**操作：** 从上一个步骤中获取数据节点的ip，在数据节点上执行`iscsiadm -m session`，从结果中解析得到对应的session ip。返回结果的格式如下tcp: [1] <ip>:<port>,xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx**需要执行的命令：**```bash\niscsiadm -m session\n```\n\n4. 检查数据节点上的网络连通性\n\n**目标：**检查数据节点到磁阵的session ip是否连通。**操作：** 遍历上一步骤中的到的ip地址，执行ping来验证网络连通性。如果执行结果包含'100% packet loss'，说明是网络问题导致的卷挂载失败。**需要执行的命令：**```bash\nping -c 3 <ip>\n```
```

<Output>
entity{tuple_delimiter}Pod_ContainerCreating_[all targetPortals not available]{tuple_delimiter}FaultSymptom{tuple_delimiter}Pod 处于 ContainerCreating 状态, Event reason 提示挂卷失败报错all targetPortals not available。
entity{tuple_delimiter}数据节点与后端存储ip网络不通{tuple_delimiter}RootCause{tuple_delimiter}因为节点与网络不通，卷挂载失败，并最终导致的Pod处于ContainerCreating阶段
entity{tuple_delimiter}网络故障导致的Pod ContainerCreating故障{tuple_delimiter}FaultCategory{tuple_delimiter}因为网络不通，导致Pod 处于ContainerCreating，无法正常启动。
entity{tuple_delimiter}获取Pod对应的数据节点信息{tuple_delimiter}CheckStep{tuple_delimiter}获取故障Pod所在的数据节点信息和node_ip，从上一步Pod JSON中提取nodename并查询节点详情
entity{tuple_delimiter}在数据节点上获取磁阵的ip地址{tuple_delimiter}CheckStep{tuple_delimiter}获取节点上磁阵的ip session，在数据节点上执行iscsiadm命令解析session ip
entity{tuple_delimiter}检查数据节点上的网络连通性{tuple_delimiter}CheckStep{tuple_delimiter}检查数据节点到磁阵的session ip是否连通，使用ping验证网络连通性，判断是否丢包
entity{tuple_delimiter}kubectl get pod{tuple_delimiter}CodeAction{tuple_delimiter}kubectl get pod <pod_name> -n <namespace> -o json
entity{tuple_delimiter}kubectl get events{tuple_delimiter}CodeAction{tuple_delimiter}kubectl get events --field-selector=involvedObject.name=<pod_name> -n <namespace> --sort-by='.lastTimestamp' -o json
entity{tuple_delimiter}iscsiadm session{tuple_delimiter}CodeAction{tuple_delimiter}iscsiadm -m session
entity{tuple_delimiter}ping storage ip{tuple_delimiter}CodeAction{tuple_delimiter}ping -c 3 <ip>
entity{tuple_delimiter}网络不通导致卷挂载失败{tuple_delimiter}Conclusion{tuple_delimiter}执行结果包含'100% packet loss'，确认是网络问题导致的卷挂载失败，all targetPortals not available根因确认
relation{tuple_delimiter}Pod Fault: all targetPortals not available{tuple_delimiter}Pod ContainerCreating故障{tuple_delimiter}BELONGS_TO{tuple_delimiter}Pod挂卷失败现象属于ContainerCreating故障分类
relation{tuple_delimiter}Pod ContainerCreating故障{tuple_delimiter}网络不通{tuple_delimiter}HAS_ROOT_CAUSE{tuple_delimiter}ContainerCreating故障可能由网络不通导致存储无法挂载引起
relation{tuple_delimiter}网络不通{tuple_delimiter}获取故障 Pod 信息和事件{tuple_delimiter}REQUIRES_STEP{tuple_delimiter}排查网络不通根因需要获取故障Pod信息和事件
relation{tuple_delimiter}网络不通{tuple_delimiter}获取Pod对应的数据节点信息{tuple_delimiter}REQUIRES_STEP{tuple_delimiter}排查网络不通根因需要获取Pod对应的数据节点信息
relation{tuple_delimiter}网络不通{tuple_delimiter}在数据节点上获取磁阵的ip地址{tuple_delimiter}REQUIRES_STEP{tuple_delimiter}排查网络不通根因需要获取磁阵IP地址
relation{tuple_delimiter}网络不通{tuple_delimiter}检查数据节点上的网络连通性{tuple_delimiter}REQUIRES_STEP{tuple_delimiter}排查网络不通根因需要检查数据节点上的网络连通性
relation{tuple_delimiter}获取故障 Pod 信息和事件{tuple_delimiter}kubectl get pod{tuple_delimiter}USES_CODE{tuple_delimiter}获取Pod信息需要执行kubectl get pod命令
relation{tuple_delimiter}获取故障 Pod 信息和事件{tuple_delimiter}kubectl get events{tuple_delimiter}USES_CODE{tuple_delimiter}获取相关事件需要执行kubectl get events命令
relation{tuple_delimiter}获取故障 Pod 信息和事件{tuple_delimiter}获取Pod对应的数据节点信息{tuple_delimiter}NEXT_STEP{tuple_delimiter}获取Pod信息后下一步是获取数据节点信息
relation{tuple_delimiter}获取Pod对应的数据节点信息{tuple_delimiter}在数据节点上获取磁阵的ip地址{tuple_delimiter}NEXT_STEP{tuple_delimiter}获取节点信息后下一步是获取磁阵IP地址
relation{tuple_delimiter}在数据节点上获取磁阵的ip地址{tuple_delimiter}iscsiadm session{tuple_delimiter}USES_CODE{tuple_delimiter}获取磁阵session需要执行iscsiadm -m session命令
relation{tuple_delimiter}在数据节点上获取磁阵的ip地址{tuple_delimiter}检查数据节点上的网络连通性{tuple_delimiter}NEXT_STEP{tuple_delimiter}获取磁阵IP后下一步是检查网络连通性
relation{tuple_delimiter}检查数据节点上的网络连通性{tuple_delimiter}ping storage ip{tuple_delimiter}USES_CODE{tuple_delimiter}检查网络连通性需要执行ping命令测试
relation{tuple_delimiter}检查数据节点上的网络连通性{tuple_delimiter}网络不通导致卷挂载失败{tuple_delimiter}LEADS_TO{tuple_delimiter}检查网络连通性后得出网络不通导致卷挂载失败的结论
{completion_delimiter}
""",
]

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.
6. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
  - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
  - If conflicts within a single entity/relation (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
7. Length Constraint:The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
8. Language: The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) may in their original language if proper translation is not available.
  - The entire output must be written in {language}.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""

PROMPTS["naive_rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

PROMPTS["kg_query_context"] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["naive_query_context"] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.
5. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}

""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}

""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}

""",
]
