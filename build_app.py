import joblib, json
p = joblib.load("churn_outputs/churn_model.joblib")
pre, clf = p.named_steps["pre"], p.named_steps["clf"]
sc = pre.named_transformers_["num"]; oh = pre.named_transformers_["cat"]
num_cols = pre.transformers_[0][2]; cat_cols = pre.transformers_[1][2]
coef = clf.coef_[0]; i = 0; M = {"b": float(clf.intercept_[0]), "num": {}, "cat": {}}
for j, c in enumerate(num_cols):
    M["num"][c] = [float(coef[i]), float(sc.mean_[j]), float(sc.scale_[j])]; i += 1
for c, cats in zip(cat_cols, oh.categories_):
    M["cat"][c] = {str(k): float(coef[i + n]) for n, k in enumerate(cats)}; i += len(cats)
open("model.json", "w").write(json.dumps(M))
html = open("template.html").read().replace("__MODEL__", json.dumps(M, separators=(",", ":")))
open("churn_app.html", "w").write(html)
print(len(html), "bytes")
