# -----------------------------------------------------------------------------
# Copyright (c) 2015, Nicolas P. Rougier. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
#import datetime
from io import StringIO

#n = 1024
#X = np.random.normal(0,1,n)
#Y = np.random.normal(0,1,n)
#T = np.arctan2(Y,X)



#https://docs.google.com/spreadsheets/d/1ukgxN4wSTy05JaxdcMwC96OygQX6Fud1QJzulK7cjJE#gid=1260460495&output=csv
#https://docs.google.com/spreadsheets/d/1WPNBlkhMsqUhrvOL6N3kkQpED8XheDG70p3MDFiFTb8/gviz/tq?gid=2115717185

r=requests.get('https://docs.google.com/spreadsheets/d/1ukgxN4wSTy05JaxdcMwC96OygQX6Fud1QJzulK7cjJE/export?format=csv&gid=1260460495&q=SELECT C')
df=pd.read_csv(StringIO(r.content.decode('utf-8')),index_col=2)
df.columns=df.iloc[0]
df=df.reindex(df.index.drop('USN'))
df=df[df.columns[2:]]
df['STARTED']=pd.to_datetime(df['STARTED'])
df['ENDED']=pd.to_datetime(df['ENDED'])
df['DAYS']=pd.to_numeric(df['DAYS'])
df['SP']=pd.to_numeric(df['SP'])
df['ISO_WEEK']=[x.isocalendar()[1] for x in list(df['ENDED'])]

#Primeras 4 semanas x mes:
dfmes=df[df['ISO_WEEK']<df['ISO_WEEK'].min()+4]

X=dfmes['ENDED']
Y=dfmes['DAYS']
T=dfmes['SP']
S=dfmes['ISO_WEEK']

ax=plt.axes()
for i in range(dfmes.index.values.size) :
    ax.annotate(dfmes.index.values[i],dfmes[['ENDED','DAYS']].values[i])

dfsem=pd.DataFrame(

        [
            (x,dfmes[dfmes['ISO_WEEK']==x]['ISO_WEEK'].count(), dfmes[dfmes['ISO_WEEK']==x]['DAYS'].mean(), dfmes[dfmes['ISO_WEEK']==x]['DAYS'].std())
            for x in dfmes['ISO_WEEK'].drop_duplicates().values
        ],columns=['WEEK','DELIVERS','MEAN_DAYS','STD_DAYS']
)
dfsem.index=dfsem['WEEK']
dfsem.drop('WEEK',axis=1)


plt.subplot(211)
plt.plot(range(12))
plt.subplot(212)

#plt.axes([0.025,0.025,0.95,0.95])
plt.scatter(X.values,Y.values, s=75, c=T.values, cmap='Reds', alpha=1)
#plt.xlim((X.min()-datetime.timedelta(days=1),X.max()+datetime.timedelta(days=1)))
minsemday=X.min()-pd.datetools.timedelta(days=int(pd.datetime.strftime(X.min(),'%w')))
maxsemday=X.max()+(pd.datetools.timedelta(days=7-int(pd.datetime.strftime(X.max(),'%w'))))
plt.xlim(minsemday,maxsemday)
loc,labels=plt.xticks()
plt.xticks(np.linspace(pd.datetime.toordinal(minsemday),pd.datetime.toordinal(maxsemday),num=5), S.drop_duplicates().sort_values().values,rotation=90 )
#plt.xticks(np.linspace(loc.min(),loc.max(),num=5), S.drop_duplicates().sort_values().values,rotation=90 )
#   plt.xticks(rotation=90 )
ax.grid()

#plt.grid()
    
#plt.xlim(-1.5,1.5), plt.xticks([])
#plt.ylim(-1.5,1.5), plt.yticks([])


xpromsem=np.linspace(plt.xlim()[0],plt.xlim()[1],num=5)
ypromsem=dfsem.sort_index()['MEAN_DAYS'].values
#yvals=np.repeat(yvals,2)
plt.plot(xpromsem[0:2],(ypromsem[0],ypromsem[0]),'r--')
plt.plot(xpromsem[1:3],(ypromsem[1],ypromsem[1]),'r--')
plt.plot(xpromsem[2:4],(ypromsem[2],ypromsem[2]),'r--')
plt.plot(xpromsem[3:5],(ypromsem[3],ypromsem[3]),'r--')




# savefig('../figures/scatter_ex.png',dpi=48)
#plt.show()