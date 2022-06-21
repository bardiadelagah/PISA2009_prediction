# PISA2009_prediction

An SGDregressor prediction model with PISA2009 dataset

# How it works

In this python code, we predict students' grades based on some of their characteristics. To do this, we want to use linear regression. Here we use the SGDregressor 
model in the model_linear module on the scikit library. We set the following values for the stated parameters and find the best value for the learning rate (eta0) 
among the following values using the GridSearchCV and 5fold cross-validation method. 

learning_rate: 'adaptive' 

max_iter : 1000

After finding the best hyperparameters, we fit the model.

After finding the best hyperparameters, we fit the model. We use a dataset called PISA2009, which is not a scikit-ready data set and must be read from an external file. The list below shows the 23 columns as features and the 24th column as the label of the model output. Below are all 23 features along with their descriptions.

grade: The grade in school of the student (most 15-year-olds in America are in 10th grade)

male: Whether the student is male (1/0)

raceeth: The race/ethnicity composite of the student

preschool: Whether the student attended preschool (1/0)

expectBachelors: Whether the student expects to obtain a bachelor's degree (1/0)

motherHS: Whether the student's mother completed high school (1/0)

motherBachelors: Whether the student's mother obtained a bachelor's degree (1/0)

motherWork: Whether the student's mother has part-time or full-time work (1/0)

fatherHS: Whether the student's father completed high school (1/0)

fatherBachelors: Whether the student's father obtained a bachelor's degree (1/0)

fatherWork: Whether the student's father has part-time or full-time work (1/0)

selfBornUS: Whether the student was born in the United States of America (1/0)

motherBornUS: Whether the student's mother was born in the United States of America (1/0)

fatherBornUS: Whether the student's father was born in the United States of America (1/0)

englishAtHome: Whether the student speaks English at home (1/0)

computerForSchoolwork: Whether the student has access to a computer for schoolwork (1/0)

read30MinsADay: Whether the student reads for pleasure for 30 minutes/day (1/0)

minutesPerWeekEnglish: The number of minutes per week the student spend in English class

studentsInEnglish: The number of students in this student's English class at school

schoolHasLibrary: Whether this student's school has a library (1/0)

publicSchool: Whether this student attends a public school (1/0)

urban: Whether this student's school is in an urban area (1/0)

schoolSize: The number of students in this student's school

readingScore: The student's reading score, on a 1000-point scale


## Donate us
If you like our project and it's useful, feel free to donate us.

Bitcoin(BTC): bc1qs2fatdfdvc5jyq4a0f5t7plmy8sxmyk08tq5e5

Ethereum(ETH): 0x5847D46Bfed82a475ef4187cfBD55EF412C05093

Tether(USDT-TRC20): TAmbZwJXDZ8bo2hjGXtNkTSEYi8dt2Xww8

XRP(XRP): rqTpCtGtBEhcPjZLXfNTv3JbCdkRKGLCF

Dogecoin(DOGE): DGZYMS6nnT3cBYwDtSD7VVubr1dSfykURC

TRON(TRX): TAmbZwJXDZ8bo2hjGXtNkTSEYi8dt2Xww8

BitTorrent-New(BTT-BEP20): 0x5847D46Bfed82a475ef4187cfBD55EF412C05093

Decentraland(MANA-ERC20): 0x5847D46Bfed82a475ef4187cfBD55EF412C05093

