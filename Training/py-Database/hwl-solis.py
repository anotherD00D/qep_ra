import pandas as pd
from contextlib import redirect_stdout

def processFile(file_name:str):
    raw_read = pd.read_csv(file_name, header=None, names=['raw'])
    raw_df = raw_read['raw'].str.split(n=5, expand=True)
    raw_df.columns = ['Student ID', 'First Name', 'Last Name', 'Major', 'Course Count', 'Courses']

    student_df, courses_df, GPA_df = createStudentDataframes(raw_df)

    return student_df, courses_df, GPA_df

def createStudentDataframes(raw_df:pd.DataFrame):
    student_df = raw_df[['Student ID', 'First Name', 'Last Name', 'Major', 'Course Count']]

    filtered_df = raw_df[['Student ID', 'Courses']].copy()
    regex_pattern = r'(\S+)\s+(\d+)\s+(\d+)\s+(\S+)'
    filtered_df['Courses'] = filtered_df["Courses"].fillna("").str.findall(regex_pattern)
    exploded_df = filtered_df.explode('Courses').dropna(subset='Courses').reset_index(drop=True)

    courses_df = pd.concat([exploded_df[["Student ID"]], exploded_df["Courses"].apply(pd.Series)], axis=1
                               ).rename(columns={0: "Department", 1: "Number", 2: "Credits", 3: "Grade"})
    
    courses_df['Credits'] = courses_df['Credits'].astype(int)
    courses_df['Number'] = courses_df['Number'].astype(int)

    GPA_df = getGPADataFrame(student_df, courses_df)

    return student_df, courses_df, GPA_df

def getGPADataFrame(student_df:pd.DataFrame, courses_df:pd.DataFrame):
    gpa_dict = {"A":4.0, "A-":3.7,
                "B+":3.3, "B":3.0, "B-":2.7,
                "C+":2.3, "C":2.0, "C-":1.7,
                "D+":1.3, "D":1.0, "D-":0.7,
                "F":0.0}
    new_row = []

    for idx_outer, row_outer in student_df.iterrows():
        total_hours = 0
        quality_hours = 0
        
        for idx_inner, row_inner in courses_df[courses_df['Student ID'] == row_outer['Student ID']].iterrows():
            total_hours += row_inner['Credits']
            quality_hours += gpa_dict[row_inner['Grade']] * row_inner['Credits']
        
        if total_hours == 0:
            GPA = 0
        else:
            GPA = round(quality_hours/total_hours, ndigits=2)
        
        new_row.append({'Student ID': row_outer['Student ID'], 'GPA': GPA})
    
    GPA_df = pd.DataFrame(new_row, columns=['Student ID', 'GPA'])

    return GPA_df

def createStudentDict(student_df:pd.DataFrame, GPA_df:pd.DataFrame):
    df = pd.merge(student_df[['Student ID','Major']], GPA_df, on='Student ID')
    student_dict = df.set_index('Student ID').to_dict(orient="index")
    student_dict = {outer_key: tuple(inner_dict.values())
                    for outer_key, inner_dict in student_dict.items()}
    
    return student_dict

def createRoster(student_df:pd.DataFrame, courses_df:pd.DataFrame, department:str, course_number:int):
    filter_mask = courses_df['Department'].str.contains(department) & (courses_df['Number'] == course_number)

    grades_df = courses_df.loc[filter_mask, ['Student ID', 'Grade']]
    student_df = student_df[['Student ID', 'First Name', 'Last Name']]

    df = student_df.merge(grades_df, on='Student ID')

    output_list = list(df.itertuples(index=False, name=None))

    return output_list

def createCourseSet(courses_df:pd.DataFrame, department):
    filter_mask = courses_df['Department'].str.contains(department)

    courses_df = courses_df.loc[filter_mask, ['Department', 'Number']]
    output_set = {f'{course_row['Department']} {course_row['Number']}'
                  for idx, course_row in courses_df.iterrows()}

    return output_set

def printStudents(student_df:pd.DataFrame, GPA_df:pd.DataFrame):

    df = student_df.merge(GPA_df, on='Student ID')
    df = df[['Student ID', 'First Name', 'Last Name', 'Major', 'GPA']].sort_values(by='Student ID').reset_index(drop=True)
    print(df)
    save([df], file_prefix, output_file)

def addStudent(file_name:str, id_number:int, first_name:str, last_name:str, major:str, number:int, courses:list):
    number = 2
    write_line = f'{id_number} {first_name} {last_name} {major} {number}'
    for single_course in courses:
        write_line += f' {single_course[0]} {single_course[1]} {single_course[2]} {single_course[3]}'
    
    with open(f"{file_name}", "a") as f:
        f.write(f"\n{write_line}")
    return -1

def save(value_list:list, file_prefix, file_name):
    out_path = rf"{file_prefix}\{file_name}"
    fill_in = '='
    for value in value_list:
        with open(out_path, 'a') as out, redirect_stdout(out):
            print(value, end=f"\n\n{fill_in*100}\n\n")

if __name__ == '__main__':
    isTest = True
    file_prefix = r'Training\py-Database'

    #Step 1:
    if isTest:
        user_name = 'Juan Solis'
        input_file, output_file = r"studentsDB.txt", r"output_file.txt"
    else:
        user_name = input("Please Enter User Name")
        input_file, output_file = input("Provide the file name of the input file:"), input("Provide the file name of the output file:")

    #Step 2:
    addStudent(rf'{file_prefix}\{input_file}', 7444, 'Khaled', 'Enab', 'PE', 2, [('CSCE', 1336, 5, 'D-'),('PE', 3300, 3, 'A')])
    addStudent(rf'{file_prefix}\{input_file}', 5564, 'Elvira', 'Teran', 'CSCE', 1, [('CSCE', 1336, 5, 'A')])

    #Step 3:
    save([f"User Name:{user_name}",f"Input File:{input_file}"], file_prefix, output_file)

    #Step 4:
    student_df, courses_df, GPA_df = processFile(rf'{file_prefix}\{input_file}')

    #Step 5:
    save([student_df, courses_df, GPA_df], file_prefix, output_file)
    
    #Step 6:
    student_dict = createStudentDict(student_df, GPA_df)
    
    #Step 7:
    save([student_dict], file_prefix, output_file)
    
    #Step 8:
    roster_set = createRoster(student_df, courses_df, "CSCE", 1336)
    
    #Step 9:
    save([roster_set], file_prefix, output_file)
    
    #Step 10:
    roster_set = createRoster(student_df, courses_df, "Math", 2340)
    
    #Step 11:
    save([roster_set], file_prefix, output_file)

    #Step 12:
    course_set = createCourseSet(courses_df, "CSCE")
    
    #Step 13:
    save([course_set], file_prefix, output_file)

    #Step 14:
    course_set = createCourseSet(courses_df, "Math")
    
    #Step 15:
    save([course_set], file_prefix, output_file)

    #Step 16:
    printStudents(student_df, GPA_df)


        
    

    
