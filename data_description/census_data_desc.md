
## Adult Dataset Information
Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

Prediction task is to determine whether a person makes over 50K a year.

## Attribute Information
| attribute| Category |
|  ----  | ---- |
| label| >50K, <=50K.|
| age  | continuous |
| fnlwgt  | continuous |
| education-num  | continuous |
| capital-gain  | continuous |
| capital-loss  | continuous |
| hours-per-week  | continuous |
| sex  | Female, Male. |
| race  |  White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. |
| occupation  |  Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. |
| relationship  |  Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. |
| marital-status  |  Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. |
| workclass  | Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. |
| education  | Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
| native-country  | United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.|


## Relevant Papers
[1]. Ron Kohavi, ["Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid"](http://robotics.stanford.edu/~ronnyk/nbtree.pdf]), Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996.

