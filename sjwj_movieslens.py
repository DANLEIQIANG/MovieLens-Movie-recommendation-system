import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import *
import warnings
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ratingsPath = 'G:/ml-latest-small/ratings.csv'
moviesPath='G:/ml-latest-small/movies.csv'
moviesDF = pd.read_csv(moviesPath,index_col=None)     #读入movies
ratingsDF = pd.read_csv(ratingsPath,index_col=None)
rateDF=ratingsDF[ratingsDF.rating>3]       #若评分大于3则表示喜欢，在后面算入二分结构
hateDF=ratingsDF[ratingsDF.rating<=3]


trainDF,testDF = train_test_split(rateDF,test_size=0.1)  #分出训练集，测试集（再所有喜欢的电影中）算R需要
hatetrainDF,hatetestDF=train_test_split(hateDF,test_size=0.1)   #在所有不喜欢的电影中选出10%作为测试集，为了画ROC曲线

ratingsPivotDF = pd.pivot_table(trainDF[['userId','movieId','rating']],columns=['movieId'],index=['userId'],values='rating',fill_value=0)  #转化为透视表，用0填充
A=pd.pivot_table(trainDF[['userId','movieId','rating']],columns=['movieId'],index=['userId'],values='rating',fill_value=0)

moviesMap = dict(enumerate(list(ratingsPivotDF.columns)))
usersMap = dict(enumerate(list(ratingsPivotDF.index)))

ratingValues = ratingsPivotDF.values.tolist()
#print(ratingValues)

def giveScore(A):   #建立2部图
    A[A > 3] = 1
    A[A != 1] = 0
    return A

def weight(A,ratingsPivotDF):
    m,n=A.shape
    KL=n-(ratingsPivotDF==0).astype(int).sum(axis=1)  #kl表示用户l的度（用户选择过多少产品）
    KJ=m-(ratingsPivotDF==0).astype(int).sum(axis=0)  #kj表示产品j的度（被多少用户评价过）
    AA=array(A)

    temp=zeros((m,n), dtype=float)
    temp2=zeros((m,n),dtype=float)
    for i in range(0,n,1):    #计算资源配额矩阵 这里参考了网上的思路 用矩阵运算代替循环来实现 大大的提高了效率
        temp[:,i]=AA[:,i]/KL
    for j in range(0,m,1):
        temp2[j,:]=temp[j,:]/KJ
    D=dot(AA.T,temp2)
    return D

def recomend(D,ratingsPivotDF):
    ratingsPivotDF[ratingsPivotDF!=0]=1    #建立2部图，这个其实和A矩阵相同
    ratingsPivotDF[ratingsPivotDF==0]=0
    trainit=array(ratingsPivotDF)
    f=dot(D,trainit.T)     #  f’=Df
    f=f.T
    ratingsPivotDF[ratingsPivotDF==1]=2  #将之前的二部图的喜欢变为0，不喜欢变为1，这一步的目的是在推荐电影时不推荐用户已经看过的电影
    ratingsPivotDF[ratingsPivotDF==0]=1
    ratingsPivotDF[ratingsPivotDF==2]=0

    ff=multiply(f,ratingsPivotDF)  #见用户看过的电影的推荐值都改为0
    ff=array(ff)

    userRecommendDict = dict()
    for i in range(len(ratingValues)):
        userRecommendDict[i] = sorted(enumerate(list(ff[i])), key=lambda x: x[1], reverse=True)   #对用户没看过的电影进行排序

    userRecommendList = []
    #print(userRecommendDict.values())
    for key, value in userRecommendDict.items():   #这些操作都是为了输出
        user = usersMap[key]
        for (movieId, val) in value:
            userRecommendList.append([user,moviesMap[movieId],val])
    recommendDF=pd.DataFrame(userRecommendList,columns=['userId','movieId','val'])

    #recommendDFA = pd.DataFrame(userRecommendList,columns=['userId','movieId','val'])
    recommendDF = pd.merge(recommendDF,moviesDF[['movieId','title']],how='left',on='movieId')  #将推荐表和电影表连接，这样可以输出电影名称
    recommendDFA=recommendDF.drop(['val'],axis=1)  #输出的时候不输出val，看着简洁
    print(recommendDFA[recommendDFA['userId'] == 1].head(10))   #输出用户1推荐的10部电影作为样例


    return recommendDF

def test(recommendDF,testDF):
    means=0.0
    testRocscore = pd.DataFrame()

    for i in range(1,len(set(rateDF['userId'].values.tolist()))+1):   #对每个用户而言
        movieID=testDF[testDF['userId'] == i]     #每个用户用于测试的电影
        temp=[0]*len(movieID)
        ave=[0]*len(movieID)
        movieID = array(movieID['movieId'])
        recommend = recommendDF[recommendDF['userId'] == i]
        KL =len(set(rateDF['movieId'].values.tolist()))-( ratingsPivotDF == 0).astype(int).sum(axis=1)  #未选择的电影数目
        for j in range(0,len(movieID)) :
            temp[j]=where(recommend.movieId==movieID[j])[0]    #找出测试电影推荐电影中排名
            if temp[j]!=0:
                ave[j]=float(temp[j]/KL[i])

        if nanmean(ave)<3:      #有的电影不在推荐电影中，排除这些情况
            means=means+float(nanmean(ave))

    print('r='+str(means/len(set(testDF['userId'].values.tolist()))))   #计算出r


    testDF = testDF.append(hatetestDF)   #为了画roc曲线，加上用户不喜欢的电影 这里喜欢大约6000多部，不喜欢3000多部
    for i in range(1, len(set(rateDF['userId'].values.tolist())) + 1):

        movieID = testDF[testDF['userId'] == i]

        movieID = array(movieID['movieId'])
        recommend = recommendDF[recommendDF['userId'] == i]

        for j in range(0, len(movieID)):   #测出每个用户对每部电影的测试集评分
            testRocscore = testRocscore.append(recommend.loc[recommend.movieId == movieID[j]])

    testRocscore = testRocscore.reset_index(drop=True)

    testDF.loc[testDF['rating']>3, 'rating']=10    #测试集结果，将大于3分的置为1，表示喜欢，小于3的置为0，表示不喜欢
    testDF.loc[testDF['rating'] !=10, 'rating'] = 0
    testDF.loc[testDF['rating'] == 10, 'rating'] = 1
    testDF = testDF.reset_index(drop=True)

    result = pd.merge(testDF[['movieId', 'userId','rating']], testRocscore[['movieId', 'userId','val']], how='right', on=('movieId','userId'))
    score=array(result['val'])
    testtrue=array(result['rating'])



    fpr, tpr, threshold = roc_curve(testtrue,score )  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    print('auc='+str(roc_auc))

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.show()








A=giveScore(A)
D=weight(A,ratingsPivotDF)
recommendDF=recomend(D,ratingsPivotDF)
test(recommendDF,testDF)