"""阶段一：用教师模型根据法规条文出题。

用法:
    # 只看切块和 prompt，不调用模型
    uv run python gen_qa.py --dry-run --n 3
    # 正式生成（任何 OpenAI 兼容接口：DeepSeek / Qwen / vLLM 本地服务等）
    set OPENAI_API_KEY=... & set OPENAI_BASE_URL=... & set TEACHER_MODEL=...
    uv run python gen_qa.py --n 50 --out data/qa_raw.jsonl
"""
import argparse
import json
import os
import random
import re

import pandas as pd

PARQUET = "data/train-00000-of-00001.parquet"
ARTICLE_RE = re.compile(r"(?m)^(第[一二三四五六七八九十百千零]+条)")

# 出题需求：train 用于后续 RL，heldout 不参与训练，用来测泛化
REQUIREMENTS = {
    "train": [
        "出 2 道单项选择题（4 个选项），考查对条文中具体规定的理解，干扰项要似是而非。",
        "出 1 道案例分析题：构造一个贴近生活的具体情境，要求根据条文判断当事人的权利义务或法律后果。",
        "出 3 道判断题，至少 1 道为错误表述，错误点应是对条文的细微改动（如主体、期限、条件）。",
        "出 2 道简答题，考查条文中的概念定义、适用条件或程序要求。",
    ],
    "heldout": [
        "出 1 道需要综合两条及以上条文才能回答的问答题。",
        "出 2 道多项选择题（4-5 个选项，正确答案不少于 2 个）。",
    ],
}

SYSTEM_PROMPT = "你是一名资深法律考试命题专家，只依据给定的法律条文出题，不引入材料之外的法律规定。"

USER_TEMPLATE = """【参考材料】
法规名称：{title}
效力层级：{level}
条文：
{text}

【出题需求】
{requirement}

【输出要求】
只输出一个 JSON 对象，不要输出其他内容，格式如下：
{{"questions": [
  {{"type": "题型",
    "question": "题干（选择题把选项写在题干里，用 A. B. C. D. 标注）",
    "answer": "标准答案",
    "analysis": "解析，说明推理过程",
    "evidence": ["从参考材料中原样摘录的依据句子"],
    "articles": ["第X条"]}}
]}}
要求：答案必须能由参考材料唯一确定；evidence 必须是材料原文的精确摘录。"""


def split_articles(content):
    """按"第X条"切分正文，返回 [(条号, 条文文本)]。"""
    parts = ARTICLE_RE.split(content)
    return [(parts[i], (parts[i] + parts[i + 1]).strip()) for i in range(1, len(parts) - 1, 2)]


def make_chunks(df, min_chars=400, max_chars=1800):
    """把连续条文合并成长度适中的材料块。"""
    chunks = []
    for row in df.itertuples():
        buf = []
        for art_no, art_text in split_articles(row.content):
            # 去掉章节标题行，只保留条文
            art_text = re.sub(r"(?m)^#+.*$", "", art_text).strip()
            buf.append((art_no, art_text))
            if sum(len(t) for _, t in buf) >= max_chars or len(buf) >= 8:
                chunks.append(_pack(row, buf))
                buf = []
        if buf and sum(len(t) for _, t in buf) >= min_chars:
            chunks.append(_pack(row, buf))
    return chunks


def _pack(row, buf):
    return {
        "doc_id": row.id,
        "title": row.title,
        "level": row.level,
        "articles": [a for a, _ in buf],
        "text": "\n\n".join(t for _, t in buf),
    }


def load_source():
    df = pd.read_parquet(PARQUET)
    return df[
        (df.status == "有效")
        & df.level.isin(["法律", "行政法规", "司法解释"])
        & (df.article_count > 0)
    ]


def rule_check(q, chunk):
    """基础规则检查，结果后续可直接转成奖励信号。"""
    issues = []
    for k in ("question", "answer", "analysis", "evidence"):
        if not q.get(k):
            issues.append(f"缺少字段 {k}")
    src = re.sub(r"\s+", "", chunk["text"])
    for ev in q.get("evidence") or []:
        if re.sub(r"\s+", "", ev) not in src:
            issues.append(f"依据非原文: {ev[:30]}")
    for a in q.get("articles") or []:
        if a not in chunk["articles"]:
            issues.append(f"引用条号不在材料中: {a}")
    return issues


def call_teacher(client, model, prompt):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="生成多少个 (材料, 需求) 样本")
    ap.add_argument("--split", default="train", choices=["train", "heldout"])
    ap.add_argument("--out", default="data/qa_raw.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    chunks = make_chunks(load_source())
    print(f"共 {len(chunks)} 个材料块")
    samples = random.sample(chunks, args.n)

    if args.dry_run:
        c = samples[0]
        print(USER_TEMPLATE.format(requirement=REQUIREMENTS[args.split][0], **c))
        return

    from openai import OpenAI

    client = OpenAI()  # 读取 OPENAI_API_KEY / OPENAI_BASE_URL
    model = os.environ["TEACHER_MODEL"]

    with open(args.out, "a", encoding="utf-8") as f:
        for i, c in enumerate(samples):
            req = random.choice(REQUIREMENTS[args.split])
            prompt = USER_TEMPLATE.format(requirement=req, **c)
            try:
                raw = call_teacher(client, model, prompt)
                questions = json.loads(raw)["questions"]
            except Exception as e:
                print(f"[{i}] 失败: {e}")
                continue
            for q in questions:
                rec = {
                    "doc_id": c["doc_id"],
                    "title": c["title"],
                    "source_articles": c["articles"],
                    "source_text": c["text"],
                    "requirement": req,
                    "split": args.split,
                    "teacher": model,
                    **q,
                    "rule_issues": rule_check(q, c),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i}] {c['title']} {c['articles'][0]}-{c['articles'][-1]}: {len(questions)} 题")


if __name__ == "__main__":
    main()
