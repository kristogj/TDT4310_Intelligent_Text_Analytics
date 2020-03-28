from nltk.sem.logic import boolean_ops, equality_preds, binding_ops, Expression

print("Legal")
boolean_ops()
equality_preds()
binding_ops()

READ_EXPRESSION = Expression.fromstring


def task3(task, doc, expr):
    print("\nTask 4{}".format(task))
    print(doc)
    print(READ_EXPRESSION(expr))


if __name__ == '__main__':
    tasks = list("abcdefghi")
    docs = ["Angus likes someone and someone likes Julia",
            "Angus loves a dog who loves him",
            "Nobody smiles at Pat",
            "Somebody coughs and sneezes",
            "Nobody coughed or sneezed",
            "Bruce loves somebody other than Bruce",
            "Nobody other than Matthew loves Pat",
            "Cyril likes everyone except for Irene",
            "Exactly one person is asleep"]
    exprs = ["exists x.(like(angus, x) & like(x, Julia))",
             "exists x.(love(angus, x) & love(x, angus))",
             "all x.(-smile(x, pat))",
             "exists x.(cough(x) & sneeze(x))",
             "all x.(-(cough(x) | sneeze(x)))",
             "exists x.(love(bruce, x) & (x != bruce))",
             "all x.( -(matthew(x) -> love(x,pat)) )",
             "all x.(like(cyril, x))",
             "exists x.(asleep(x) & all y.(asleep(y) -> x = y))"]
    for x in range(len(docs)):
        task3(tasks[x], docs[x], exprs[x])
