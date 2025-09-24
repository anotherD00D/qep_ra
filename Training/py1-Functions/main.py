def calc_AGI(wages, interest, unemployment):
    
    return -1

def get_deduction(status):
    return -1

def calc_taxable(agi, deductions):
    return -1

def calc_tax(status, taxable):
    return -1

def calc_tax_due(tax, withheld):
    return -1

if __name__ == '__main__':
    wages, interest, unemployment, taxes = 0                #Wages, Taxable Interest, Unemployement Compensation, and Taxes
    status = {0: "Dependent", 1: "Single", 2: "Married"}    #Dictionary mapping for status

    agi = calc_AGI(wages, interest, unemployment)

    print(f"AGI: ${agi:,}")