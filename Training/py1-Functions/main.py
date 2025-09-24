def calc_AGI(wages, interest, unemployment):
    return abs(wages) + abs(interest) + abs(unemployment)

def get_deduction(status):
    deduction_dict = {  0: 6000,                    #Deduction mapping from status value (dependent (0) = 6000, single (1) = 12000, married (2) = 24000)
                        1: 12000, 
                        2: 24000}  
    deduction = deduction_dict.get(status, 6000)    #Searches key value in Deduction_Dict, if not found defaults to 6000)               
    return deduction

def calc_taxable(agi, deductions):
    return (agi-deductions) if (agi - deductions) > 0 else 0 

def binary_search(tax_bracket, taxable):
    low_idx, high_idx = 0, len(tax_bracket)

    while low_idx < high_idx:
        mid_idx = (high_idx + low_idx) // 2
        if taxable < tax_bracket[mid_idx]:
            high_idx = mid_idx
        else:
            low_idx = mid_idx + 1
        
    return low_idx - 1

def calc_tax(status, taxable):

    if status == 2:
        tax_bracket = [0, 20001, 80001]
        tax_base = [0, 2000, 9200]
        tax_perc = [0.1, 0.12, 0.22]

    else:
        tax_bracket = [0, 10001, 40001, 85001]
        tax_base = [0, 1000, 4600, 14500]
        tax_perc = [0.1, 0.12, 0.22, 0.24]

    idx = binary_search(tax_bracket, taxable)
    return round(tax_base[idx] + ((taxable - tax_bracket[idx]) * tax_perc[idx]))

def calc_tax_due(tax, withheld):
    return (tax - withheld) if withheld > 0 else tax

if __name__ == '__main__':
    
    isTest = True

    if isTest:
       wages, interest, unemployment, status, taxes = 80000, 0, 500, 2, 12000 
    else:
        wages, interest, unemployment, status, taxes = float(input()), float(input()), float(input()), int(input()), float(input())     #Wages, Taxable Interest, Unemployement Compensation, and Taxes
    

    agi = calc_AGI(wages, interest, unemployment)
    deduction = get_deduction(status)
    taxable = calc_taxable(agi, deduction)
    tax = calc_tax(status, taxable)
    tax_due = calc_tax_due(tax, taxes)

    print(f"AGI: ${agi:,}")
    print(f"Deduction: ${deduction:,}")
    print(f"Taxable Income: ${taxable:,}")
    print(f"Tax: ${tax:,}")
    print(f"Tax due: ${tax_due:,}")