from sklearn import tree

# [weight g, color ]
# 2 = red, 3 = yellow , 4 = orange
features = [[102,2],[102,3],[300,4],[400,4]]
labels = ['Tfa7a', 'Tfa7a', 'Limouna', 'Limouna']

clasf = tree.DecisionTreeClassifier()
classif = clasf.fit(features, labels)

print (classif.predict([[160,3]]))

