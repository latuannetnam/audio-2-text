from infer import infer

text_input = "Khi nào mà cơ thể của chúng ta có những biểu diện như thế này này, đầy bộng, nôn, quạng"
norm_result = infer([text_input], '')
print(norm_result)