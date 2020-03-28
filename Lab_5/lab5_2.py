from nltk.sem.logic import boolean_ops, equality_preds, binding_ops, Expression

print("Legal")
boolean_ops()
equality_preds()
binding_ops()

READ_EXPRESSION = Expression.fromstring


def task2(task, doc, expr, explication):
    print("\nTask 2{}".format(task))
    print(doc)
    print(READ_EXPRESSION(expr))
    print(explication)


if __name__ == '__main__':
    tasks = list("abcdef")
    docs = ["If Angus sings, it is not the case that Bertie sulks.",
            "Cyril runs and barks.",
            "It will snow if it doesn’t rain.",
            "It’s not the case that Irene will be happy if Olive or Tofu comes.",
            "Pat didn’t cough or sneeze.",
            "If you don’t come if I call, I won’t come if you call."]
    exprs = ["A -> -B",
             "R & B",
             "-R -> S",
             "(O | T) -> -IH",
             "-(C | S)",
             "(A -> -B) -> (C -> -D)"]
    explinations = ["A = Angus sings, B = Bertie sulks",
                    "R = Cyril runs, B = Cyril Barks",
                    "R = It will rain, S = It will snow",
                    "O = Olive comes, T = Tofu comes, IH = Irene will be happy",
                    "C = Pat cough, S = Pat sneeze",
                    "A = I call, B = You come, C = You call, D = I come"]
    for x in range(len(tasks)):
        task2(tasks[x], docs[x], exprs[x], explinations[x])
