import pandas as pd

def processFile(file_name:str):
    raw_read = pd.read_csv(file_name, header=None, names=['raw'])
    raw_df = raw_read['raw'].str.split(n=5, expand=True)
    raw_df.columns = ['Student ID', 'First Name', 'Last Name', 'Major', 'Course Count', 'Courses']

    return raw_df

def createStudentDataframes(raw_df):
    student_df = raw_df[['Student ID', 'Major', 'Course Count']]

    filtered_df = raw_df[['Student ID', 'Courses']].copy()
    regex_pattern = r'(\S+)\s+(\d+)\s+(\d+)\s+(\S+)'
    filtered_df['Courses'] = filtered_df["Courses"].fillna("").str.findall(regex_pattern)
    exploded_df = filtered_df.explode('Courses').dropna(subset='Courses').reset_index(drop=True)

    courses_df = pd.concat([exploded_df[["Student ID"]], exploded_df["Courses"].apply(pd.Series)], axis=1
                               ).rename(columns={0: "Department", 1: "Number", 2: "Credits", 3: "Grade"})
    
    courses_df['Credits'] = courses_df['Credits'].astype(int)

    GPA_df = getGPADataFrame(student_df, courses_df)
    
    return student_df, courses_df, GPA_df

def getGPADataFrame(student_df, courses_df):
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

def createStudentDict(student_df, GPA_df):
    df = pd.merge(student_df[['Student ID','Major']], GPA_df, on='Student ID')
    student_dict = df.set_index('Student ID').to_dict(orient="index")
    student_dict = {outer_key: tuple(inner_dict.values())
                    for outer_key, inner_dict in student_dict.items()}
    return student_dict

def createRoster():
    return -1

def createCourseSet():
    return -1

def printStudents():
    return -1

def addStudent():
    return -1




if __name__ == '__main__':
    
    file_name = r"Training\py-Database\studentsDB.txt"
    raw_df = processFile(file_name)
    student_df, courses_df, GPA_df = createStudentDataframes(raw_df)
    student_dict = createStudentDict(student_df, GPA_df)
    roster_set = createRoster()



    print(courses_df)
        
    

    
