def read_status(status:str): 
    status_list = ["UG", "G", "DL"]

    if status.upper() not in status_list:
        raise ValueError("Student status value is not acceptable.  Exiting program...")
    
    return status

def calc_grade(grade:float, max:float):
    if grade > max:
        result = 100
    else:
        result = (grade/max) * 100
    return result

def calc_avg(status, grades_tuple):

    result = 0.0
    avg_matrix = {"UG":[0.20, 0.20, 0.30, 0.30],
                  "G":[0.15, 0.05, 0.35, 0.45],
                  "DL":[0.05, 0.05, 0.40, 0.50]}
    
    for i in range(4):
        result += grades_tuple[i]/100 * avg_matrix.get(status.upper())[i]
    
    return result * 100

def binary_search(arr_list, target_value):
    low_idx, high_idx = 0, len(arr_list) - 1

    while low_idx <= high_idx:
        mid_idx = (high_idx + low_idx) // 2
        if target_value >= arr_list[mid_idx]:
            low_idx = mid_idx + 1
        else:
            high_idx = mid_idx - 1
        
    return low_idx

def calc_letter(avg):
    grade_range = [60, 70, 80, 90]
    grade_letter = ['F', 'D', 'C', 'B', 'A']

    return grade_letter[binary_search(grade_range, avg)]


if __name__ == '__main__':
    HOMEWORK_MAX = 800.0
    QUIZZES_MAX = 400
    MIDTERM_MAX = 150
    FINAL_MAX = 200.0

    try:
        status = read_status(str(input()))
        in_homework, in_quiz, in_mid, in_final = map(float, input().split())

        homework_grade = calc_grade(in_homework, HOMEWORK_MAX)
        quizzes_grade = calc_grade(in_quiz, QUIZZES_MAX)
        midterm_exam = calc_grade(in_mid, MIDTERM_MAX)
        final_exam = calc_grade(in_final, FINAL_MAX)
        avg_grade = calc_avg(status, (homework_grade, quizzes_grade, midterm_exam, final_exam))

        print(f"Homework: {homework_grade:.1f}%")
        print(f"Quizzes: {quizzes_grade:.1f}%")
        print(f"Midterm: {midterm_exam:.1f}%")
        print(f"Final Exam: {final_exam:.1f}%")
        print(f"{status} average: {avg_grade:.1f}%")
        print(f"Course grade: {calc_letter(avg_grade)}")

    except ValueError as e:
        print(f"Error: {e}")