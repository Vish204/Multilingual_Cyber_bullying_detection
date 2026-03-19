from cyberbullying.explainability.shap_explainer import explain_text

text = "You are such an idiot, nobody likes you"

result = explain_text(text)

print(result)