import pandas as pd
# Dict object
courses = {'Courses':['Spark','PySpark','Java','PHP'],
 'Fee':[20000,20000,15000,10000],
 'Duration':['35days','35days','40days','30days']}
# Create DataFrame from dict
df = pd.DataFrame.from_dict(courses)
print(df)
# Create for selected columns
df = pd.DataFrame(courses, columns = ['Courses', 'Fee'])
print(df)
# Create from from_dict()
df = pd.DataFrame.from_dict(courses)
print(df)
# Dict object
courses = {'r0':['Spark',20000,'35days'],
 'r1':['PySpark',20000,'35days'],
 'r2':['Java',15000,'40days'],
 'r3':['PHP',10000,'30days'],}
columns=['Courses','Fee','Duration']
# Create from from_dict() using orient=index
df = pd.DataFrame.from_dict(courses, orient='index', columns=columns)
print(df)
# Creating from nested dictionary
courses = {'r0':{'Courses':'Spark','Fee':'20000','Duration':'35days'},
 'r1':{'Courses':'PySpark','Fee':'20000','Duration':'35days'},
 'r2':{'Courses':'Java','Fee':'15000','Duration':'40days'},
 'r3':{'Courses':'PHP','Fee':'10000','Duration':'30days'}}
df=pd.DataFrame(courses).transpose()
print(df)