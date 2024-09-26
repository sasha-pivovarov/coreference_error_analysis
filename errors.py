import json
import random
import typing as tp
from collections import Counter
from collections import defaultdict
from pathlib import Path

import jsonlines
import stanza
import tqdm

from coref_errors.alignment import GoldPredAlignment
from coref_errors.constants import ErrorType
from coref_errors.utils import bootstrap_report

if __name__ == "__main__":
    pred_path = Path("data/conll_logs/pred.json")
    gold_path = Path("data/conll_logs/gold.json")

    pred_sents = list(jsonlines.open(pred_path))
    gold_sents = list(jsonlines.open(gold_path))

    assert len(pred_sents) == len(gold_sents)
    pipe = stanza.Pipeline(lang="en", processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized=True)

    errors = {x: [] for x in ErrorType}
    total_error_counts: tp.DefaultDict[ErrorType, int] = defaultdict(int)
    ast, mst, asd, msd, est, esd = [], [], [], [], [], []
    for g, p in tqdm.tqdm(zip(gold_sents, pred_sents)):
        if g["clusters"] and p["clusters"] and g["clusters"] != p["clusters"]:
            alignment = GoldPredAlignment(g["clusters"], p["clusters"], g["cased_words"], g["deprel"], g["head2span"], g["pos"], pipe)
            for key, value in alignment.error_counts.items():
                errors[key].append(alignment)
                total_error_counts[key] += value
            missing_dep, missing_tags = alignment.missing_span_tags
            all_dep, all_tags = alignment.all_span_tags
            extra_dep, extra_tags = alignment.extra_span_tags
            ast.extend(all_tags)
            mst.extend(missing_tags)
            msd.extend(missing_dep)
            asd.extend(all_dep)
            est.extend(extra_tags)
            esd.extend(extra_dep)

    for key, value in errors.items():
        with Path(f"data/conll_logs/{key}.json").open("w") as io:
            io.writelines([x.error_line() + "\n" for x in value])

    with Path(f"data/tag_counts.json").open("w") as io:
        ast_subsample = random.sample(ast, len(mst))
        asd_subsample = random.sample(asd, len(asd))
        mst_count = dict(Counter(mst))
        ast_count = dict(Counter(ast))
        est_count = dict(Counter(est))
        esd_count = dict(Counter(esd))
        asd_count = dict(Counter(asd))
        msd_count = dict(Counter(msd))

        ms_report = bootstrap_report(ast, mst_count, len(mst), 300)
        md_report = bootstrap_report(asd, msd_count, len(msd), 300)
        es_report = bootstrap_report(ast, est_count, len(est), 300)
        ed_report = bootstrap_report(asd, esd_count, len(esd), 300)

        # # st_pairs = pair_dicts(mst, asts)
        # # sd_pairs = pair_dicts(msd, asds)
        # # st_dicts = pair_norm(mst, ast)
        # # sd_dicts = pair_norm(msd, asd)
        # # et_pairs = pair_dicts(est, asts)
        # # ed_pairs = pair_dicts(esd, asds)
        # # et_dicts = pair_norm(est, asts)
        # # ed_dicts = pair_norm(esd, asds)
        # st_pairs_norm = [normalize(x) for x in st_pairs]
        # sd_pairs_norm = [normalize(x) for x in sd_pairs]
        # et_pairs_norm = [normalize(x) for x in et_pairs]
        # ed_pairs_norm = [normalize(x) for x in ed_pairs]
        # ks_st = ks_2samp(st_pairs[0], st_pairs[1])
        # ks_sd = ks_2samp(sd_pairs[0], sd_pairs[1])
        # ks_et = ks_2samp(et_pairs[0], et_pairs[1])
        # ks_ed = ks_2samp(ed_pairs[0], ed_pairs[1])
        # print(total_error_counts)
        # print(ks_st)
        # print(ks_sd)
        # print(ks_et)
        # print(ks_ed)
        # print()
        # print(ast, mst, asd, msd)
        # print()
        # print(st_pairs_norm, sd_pairs_norm)
        # print(et_pairs_norm, ed_pairs_norm)
        # print()
        # print(sorted(st_dicts.items(), key=lambda x: x[1][0]), sorted(sd_dicts.items(), key=lambda x: x[1][0]))
        # print(sorted(et_dicts.items(), key=lambda x: x[1][0]), sorted(ed_dicts.items(), key=lambda x: x[1][0]))

        io.writelines(json.dumps([ast, mst, asd, msd, ms_report, md_report, es_report, ed_report]))


