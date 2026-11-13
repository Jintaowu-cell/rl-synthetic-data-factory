import json, os, random, re, sys
from string import Template
import pandas as pd
from openai import OpenAI

P = {k: Template(open(f"prompts/{k}.txt", encoding="utf-8").read()) for k in ("split", "qa", "judge")}
TYPES = ["单项选择题", "多项选择题", "判断题", "案例分析题", "简答题"]
WEIGHTS = {"忠实性": .25, "答案正确性": .25, "考查价值": .15, "干扰项与情境": .15, "解析质量": .10, "需求符合度": .10}
SIZES = {"sft": 200, "rl": 1000, "eval": 100}
ART = re.compile(r"(?m)^(第[一二三四五六七八九十百千零]+条)")
norm = lambda s: re.sub(r"\s", "", s)

def parse(text):  # 容忍 ```json 包裹等多余内容
    return json.loads(re.search(r"\{.*\}", text, re.S).group())

def ask(role, prompt, temperature=0.7):
    client = OpenAI(base_url=os.getenv(f"{role}_BASE_URL"), api_key=os.getenv(f"{role}_API_KEY"))
    r = client.chat.completions.create(model=os.environ[f"{role}_MODEL"], temperature=temperature,
                                       messages=[{"role": "user", "content": prompt}])
    return parse(r.choices[0].message.content)

def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def dump(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

def articles(content):  # {条号: 条文}，并截掉夹在条文后面的章节标题
    p = ART.split(content)
    return {p[i]: (p[i] + p[i + 1]).split("\n#")[0].strip() for i in range(1, len(p) - 1, 2)}

def build_tasks(n_docs=200):  # 教师切分考点；按法规划分 sft/rl/eval，比例 1:3:1
    df = pd.read_parquet("data/train-00000-of-00001.parquet")
    df = df[(df.status == "有效") & df.level.isin(["法律", "行政法规", "司法解释"]) & df.char_count.between(1000, 20000)]
    tasks = []
    for i, row in enumerate(df.sample(n_docs, random_state=0).itertuples()):
        split = ("sft", "rl", "rl", "rl", "eval")[i % 5]
        arts = articles(row.content)
        for g in ask("TEACHER", P["split"].substitute(title=row.title, text="\n\n".join(arts.values())))["groups"]:
            if src := "\n\n".join(arts[a] for a in g["articles"] if a in arts):
                tasks.append({"title": row.title, "topic": g["topic"], "source": src, "split": split})
    rng, out = random.Random(0), []
    for sp, n in SIZES.items():
        ts = [t for t in tasks if t["split"] == sp]
        out += [{**t, "id": f"{sp}-{i}", "qtype": TYPES[i % 5]} for i, t in enumerate(rng.sample(ts, min(n, len(ts))))]
    return out

def reward(q, t, judge="JUDGE"):  # 硬规则 → judge 打分 → 一票否决 → 加权得到 0–1 的奖励
    try:
        assert all(q[k] for k in ("question", "answer", "analysis", "evidence"))
        assert all(norm(e) in norm(t["source"]) for e in q["evidence"])
    except (AssertionError, KeyError, TypeError):
        return 0.0, None
    s = ask(judge, P["judge"].substitute(**t, question=json.dumps(q, ensure_ascii=False, indent=1)), 0)["scores"]
    if min(s["忠实性"]["score"], s["答案正确性"]["score"]) <= 2:
        return 0.0, s
    return sum(w * (s[k]["score"] - 1) / 4 for k, w in WEIGHTS.items()), s

def run(role, split, tag, k=1):  # 让某个角色在某个任务集上出题，并用两个 judge 分别打分
    rows = []
    for t in (t for t in load("data/tasks.jsonl") if t["split"] == split):
        for _ in range(k):
            try:
                q = ask(role, P["qa"].substitute(**t))
            except (ValueError, AttributeError):  # 输出中没有合法 JSON
                q = {}
            (r, s), (re_, _) = reward(q, t), reward(q, t, "EVAL_JUDGE")
            rows.append({**t, "qa": q, "scores": s, "reward": r, "reward_eval": re_})
    dump(rows, f"data/run_{tag}.jsonl")
    mean = lambda key: sum(key(r) for r in rows) / len(rows)
    print(f"{tag}: 训练judge {mean(lambda r: r['reward']):.3f}  评测judge {mean(lambda r: r['reward_eval']):.3f}  "
          f"平均长度 {mean(lambda r: len(json.dumps(r['qa'], ensure_ascii=False))):.0f}")

if __name__ == "__main__":
    # python pipeline.py tasks
    # python pipeline.py run <角色> <任务集> <标签> [每任务采样次数]
    if sys.argv[1] == "tasks":
        dump(build_tasks(), "data/tasks.jsonl")
    else:
        run(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 1)
