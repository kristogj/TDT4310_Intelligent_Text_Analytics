from nltk.sem.logic import boolean_ops, equality_preds, binding_ops, Expression

print("Legal")
boolean_ops()
equality_preds()
binding_ops()

READ_EXPRESSION = Expression.fromstring


def task3(task, doc, expr):
    print("\nTask 3{}".format(task))
    print(doc)
    print(READ_EXPRESSION(expr))


if __name__ == '__main__':
    tasks = list("abcdef")
    docs = ["Angus likes Cyril and Irene hates Cyril",
            "Tofu is taller than Berite",
            "Bruce loves himself and Pat does too",
            "Cyril saw Bertie, but Angus didnt",
            "Cyril is a fourlegged friend",
            "Tofu and Olive are near each other"]
    exprs = ["like(angus, cyril) & hate(irene, cyril)",
             "taller(tofu, bertie)",
             "love(bruce, bruce) & love(pat, pat)",
             "saw(cyril, bertie) and -saw(angus, bertie)",
             "fourlegged(cyril) & friend(cyril)",
             "near(tofu,olive) & near(olive,tofu)"]
    for x in range(len(docs)):
        task3(tasks[x], docs[x], exprs[x])
