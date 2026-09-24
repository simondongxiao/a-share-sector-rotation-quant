from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NORMALIZED = ROOT / "data" / "normalized"

SHENWAN_LEVEL1 = [
    "农林牧渔", "基础化工", "钢铁", "有色金属", "电子", "汽车", "家用电器", "食品饮料", "纺织服饰", "轻工制造", "医药生物", "公用事业", "交通运输", "房地产", "商贸零售", "社会服务", "银行", "非银金融", "综合", "建筑材料", "建筑装饰", "电力设备", "国防军工", "计算机", "传媒", "通信", "煤炭", "石油石化", "环保", "美容护理", "机械设备"
]

RULES = [
    ("农林牧渔", [r"农业", r"农林", r"养殖", r"畜牧", r"饲料", r"种业", r"粮食", r"渔业", r"农产品"]),
    ("基础化工", [r"化工", r"化纤", r"纯碱", r"农化", r"农药", r"化学原料", r"化学制品"]),
    ("钢铁", [r"钢铁", r"钢材"]),
    ("有色金属", [r"有色", r"稀土", r"黄金", r"铜", r"铝", r"锂", r"镍", r"钴", r"矿业"]),
    ("电子", [r"电子", r"半导体", r"芯片", r"集成电路", r"元器件", r"消费电子", r"光学光电子"]),
    ("汽车", [r"汽车", r"新能源车", r"智能车", r"汽配", r"整车"]),
    ("家用电器", [r"家电", r"白色家电", r"家用电器"]),
    ("食品饮料", [r"食品", r"饮料", r"白酒", r"酒", r"乳业", r"消费"]),
    ("纺织服饰", [r"纺织", r"服饰", r"服装"]),
    ("轻工制造", [r"轻工", r"造纸", r"家具", r"家居", r"包装"]),
    ("医药生物", [r"医药", r"医疗", r"生物", r"创新药", r"中药", r"疫苗", r"医疗器械"]),
    ("公用事业", [r"公用", r"电力", r"水务", r"燃气", r"火电", r"水电", r"核电"]),
    ("交通运输", [r"交通", r"运输", r"物流", r"港口", r"机场", r"航空", r"航运", r"高速"]),
    ("房地产", [r"房地产", r"地产", r"住房"]),
    ("商贸零售", [r"商贸", r"零售", r"商业", r"消费服务"]),
    ("社会服务", [r"社会服务", r"旅游", r"酒店", r"教育", r"服务"]),
    ("银行", [r"银行"]),
    ("非银金融", [r"证券", r"券商", r"保险", r"金融", r"非银"]),
    ("建筑材料", [r"建材", r"水泥", r"玻璃", r"建筑材料"]),
    ("建筑装饰", [r"建筑", r"装饰", r"基建", r"工程"]),
    ("电力设备", [r"电力设备", r"新能源", r"光伏", r"风电", r"储能", r"电池", r"电网", r"锂电"]),
    ("国防军工", [r"军工", r"国防", r"航空航天", r"航天", r"卫星"]),
    ("计算机", [r"计算机", r"软件", r"人工智能", r"AI", r"云计算", r"大数据", r"信创", r"数字经济"]),
    ("传媒", [r"传媒", r"影视", r"游戏", r"广告", r"出版"]),
    ("通信", [r"通信", r"5G", r"电信", r"光通信", r"数据中心"]),
    ("煤炭", [r"煤炭", r"煤"]),
    ("石油石化", [r"石油", r"石化", r"油气"]),
    ("环保", [r"环保", r"环境", r"垃圾", r"污水"]),
    ("美容护理", [r"美容", r"护理", r"化妆品"]),
    ("机械设备", [r"机械", r"工业", r"机器人", r"机床", r"装备", r"工程机械"]),
]

NON_INDUSTRY = [r"沪深", r"中证", r"上证", r"创业板", r"科创", r"国证", r"红利", r"价值", r"成长", r"低波", r"央企", r"国企", r"宽基", r"债券", r"货币", r"商品", r"原油", r"豆粕", r"QDII", r"港股", r"恒生", r"纳斯达克", r"标普", r"日经", r"德国", r"法国", r"美国", r"全球", r"REIT", r"可转债"]


def map_one(name: str, fund_type: str) -> tuple[str, str, str, str]:
    text = f"{name or ''} {fund_type or ''}"
    if re.search(r"商品型|债券型|货币型|REIT|可转债", text, flags=re.I) or re.search(r"黄金ETF|原油|豆粕", name or "", flags=re.I):
        return "", "non_industry_or_broad", "asset_type_rule", "商品、债券、货币、REIT或其他非股票行业产品"
    if any(re.search(pattern, text, flags=re.I) for pattern in NON_INDUSTRY):
        # Sector-specific rules get priority for names such as "有色金属" or "黄金股".
        if not any(label in text for label in ["行业", "板块", "主题", "产业"]):
            return "", "non_industry_or_broad", "non_industry_rule", "宽基、跨境、商品、债券、货币、REIT或其他非申万一级行业产品"
    hits = []
    for industry, patterns in RULES:
        if any(re.search(pattern, text, flags=re.I) for pattern in patterns):
            hits.append(industry)
    hits = list(dict.fromkeys(hits))
    if len(hits) == 1:
        return hits[0], "candidate", "name_keyword_rule", "候选映射；需用基金合同/跟踪指数复核"
    if len(hits) > 1:
        return "", "manual_review", "multiple_keyword_hits", ";".join(hits)
    return "", "unmapped", "no_rule_match", "需要查阅跟踪指数或基金合同"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(NORMALIZED / "etf_universe_with_tracking_review.csv"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype={"code": str})
    frame["code"] = frame["code"].str.zfill(6)
    mapped = frame.apply(lambda row: map_one(" ".join(str(row.get(col, "") or "") for col in ["fund_name", "tracking_index"]), row.get("fund_type", "")), axis=1, result_type="expand")
    mapped.columns = ["shenwan_level1", "mapping_status", "mapping_method", "mapping_note"]
    base_cols = ["code", "exchange", "fund_name", "fund_type", "unit_nav", "asset_size_cny", "size_eligible_current", "asset_size_asof", "inception_date", "tracking_index", "tracking_status"]
    out = pd.concat([frame[[c for c in base_cols if c in frame.columns]], mapped], axis=1)
    out["mapping_method"] = out.apply(lambda row: "tracking_index_keyword_rule" if pd.notna(row.get("tracking_index")) and str(row.get("tracking_index")).strip() else row["mapping_method"], axis=1)
    out["mapping_version"] = "shenwan_level1_tracking_index_keyword_draft_2026-09-23"
    out["mapped_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    out.to_csv(NORMALIZED / "etf_shenwan_level1_mapping_draft.csv", index=False, encoding="utf-8-sig")
    template = out[["code", "fund_name", "shenwan_level1", "mapping_status", "mapping_method", "mapping_note"]].copy()
    template["manual_override"] = ""
    template["manual_source"] = ""
    template["manual_checked_at"] = ""
    template.to_csv(NORMALIZED / "etf_shenwan_level1_mapping_manual_review.csv", index=False, encoding="utf-8-sig")
    summary = out.groupby(["mapping_status", "shenwan_level1"], dropna=False).size().reset_index(name="count")
    summary.to_csv(ROOT / "reports" / "mapping_coverage_summary.csv", index=False, encoding="utf-8-sig")
    print(out["mapping_status"].value_counts(dropna=False).to_string())
    print("saved", NORMALIZED / "etf_shenwan_level1_mapping_draft.csv")


if __name__ == "__main__":
    main()
