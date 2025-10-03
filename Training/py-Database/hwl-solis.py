import pandas as pd

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

def addStudent():
    return -1




if __name__ == '__main__':
    
    file_name = r"Training\py-Database\studentsDB.txt"
    student_df, courses_df, GPA_df = processFile(file_name)
    student_dict = createStudentDict(student_df, GPA_df)
    roster_set = createRoster(student_df, courses_df, "CSCE", 3390)
    course_set = createCourseSet(courses_df, "CSCE")
    printStudents(student_df, GPA_df)

        
    

    
