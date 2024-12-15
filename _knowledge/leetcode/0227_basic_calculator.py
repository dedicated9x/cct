class Product:
    def __init__(self, sign, s):
        self.sign = sign
        self.s = s

    def evaluate(self) -> int:
        list_signs = ['*'] + [e for e in self.s if e in ['*', '/']]
        list_factors = self.s.replace("*", "@").replace("/", "@").split("@")
        val = 1
        for sign, factor in zip(list_signs, list_factors):
            factor_int = int(factor)
            if sign == "*":
                val = val * factor_int
            else:
                val = int(val / factor_int)

        if self.sign == "+":
            pass
        else:
            val = -val

        return val


    def __repr__(self):
        return f"{self.sign} {self.s}"

class Solution:
    def calculate(self, s: str) -> int:
        # Remove spaces
        s = s.replace(" ", "")

        list_signs = ['+'] + [e for e in s if e in ['+', '-']]
        list_strings_products = s.replace("+", "@").replace("-", "@").split("@")
        list_products = [Product(sign, s) for sign, s in zip(list_signs, list_strings_products)]

        list_summands = []
        for p in list_products:
            summand = p.evaluate()
            list_summands.append(summand)

        retval = sum(list_summands)
        return retval


print(Solution().calculate(s="3 + 2*2-3+2-5*7/ 3"))