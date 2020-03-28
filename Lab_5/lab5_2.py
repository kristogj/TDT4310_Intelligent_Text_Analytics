import nltk
from nltk.sem.logic import LogicParser, boolean_ops, equality_preds, binding_ops, Expression
from nltk.sem.relextract import extract_rels
import re

print("Legal")
boolean_ops()
equality_preds()
binding_ops()

READ_EXPRESSION = Expression.fromstring


def task2(task, doc, expr):
    print("Task 2{}".format(task))
    print(doc)
    print(READ_EXPRESSION(expr))


if __name__ == '__main__':
    tasks = list("abcdef")
    docs = ["If Angus sings, it is not the case that Bertie sulks.",
            "Cyril runs and barks.",
            "It will snow if it doesn’t rain.",
            "It’s not the case that Irene will be happy if Olive or Tofu comes.",
            "Pat didn’t cough or sneeze.",
            "If you don’t come if I call, I won’t come if you call."]
    exprs = ["A -> -B",
             "RUN & BARK",
             "-RAIN -> SNOW",
             "(OLIVE | TOFU) -> -HAPPY",
             "-(COUGH | SNEEZE)",
             "D"]
    for x in range(len(tasks)):
        task2(tasks[x], docs[x], exprs[x])
