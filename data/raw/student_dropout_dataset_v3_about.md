About Dataset
Description
This dataset simulates a realistic academic environment for 10,000 students and is designed for predicting student dropout. It includes demographic, behavioral, and academic performance features such as GPA, semester GPA, CGPA, study habits, attendance, stress index, parental education, and department.

Although synthetic, the data closely mimics real-world student distributions, including slight skewness in income and stress, missing values (~5%) in some features, and logical correlations between academic performance and dropout risk.

Target variable:

Dropout (0 = retained, 1 = dropped out)
Key Features:

Student_ID: Unique student identifier
Age: Student age in years
Gender: Male / Female
Family_Income: Monthly family income
Internet_Access: Yes / No
Study_Hours_per_Day: Average study hours per day
Attendance_Rate: Percentage of attendance
Assignment_Delay_Days: Average assignment delay in days
Travel_Time_Minutes: Daily commute in minutes
Part_Time_Job: Yes / No
Scholarship: Yes / No
Stress_Index: Self-reported stress level (1–10)
GPA, Semester_GPA, CGPA: Academic performance
Semester: Current year (Year 1–4)
Department: Science, Arts, Business, CS, Engineering
Parental_Education: Highest education of parents
Dataset Characteristics:

Rows: 10,000
Columns: 19
Dropout rate: 23.5%
Contains both categorical and numerical features
Missing values are present in 5–5.1% of some columns (realistic for educational datasets)
Potential Use Cases:

Supervised machine learning classification (predict dropout)
Educational analytics and student performance research
Feature engineering challenges using derived metrics (e.g., Overloaded_Semester, Income_per_Travel_Time)
Benchmarking ML algorithms on tabular and slightly imbalanced datasets
Baseline Model:

Logistic Regression ROC-AUC: 0.818
Most predictive features: GPA, CGPA, Semester_GPA, Attendance_Rate, Stress_Index