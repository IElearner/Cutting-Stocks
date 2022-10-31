import numpy as np
import gurobipy as gp
from gurobipy import GRB


class Master:      #主问题一个lp问题
    def __init__(self, lengths, demands, W) -> None:#lengths各所需长度，demands各长度需求数量，w一根木头长度
        self.M, self.lengths, self.demands, self.W = len(lengths), lengths, demands, W#M是长度种类数量
        self.n_col, self.n_dim = 0, 0

    def create_model(self):
        self.x = []
        self.model = gp.Model("Master")
        self.__set_vars()     #定义变量（函数见下）
        self.__set_contrs()   #添加约束

    def solve(self, flag=0):
        self.model.Params.OutputFlag = flag
        self.model.optimize()

# getAttr()用于查看gurobi参数属性，Pi属性是当前解的约束对偶值（即影子价格），返回每个约束对应的对偶值
    def get_dual_vars(self):
        return [self.constrs[i].getAttr(GRB.Attr.Pi) for i in range(len(self.constrs))]

#x[i]方案对应一条木头尽可能切一种长度方案，总共M种（初始列？）
    def __set_contrs(self) -> None:
        self.constrs = self.model.addConstrs(
            (self.x[i] * (self.W // self.lengths[i]) >= self.demands[i]) for i in range(self.M))

#设置变量，lb变量下限，ub上限（无穷），vtype变量类型（连续），变量名字xi （str将变量转换成字符串）
    def __set_vars(self) -> None:
        for i in range(self.M):
            self.x.append(self.model.addVar(obj=1, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='x' + str(i)))
        self.n_col = 1        #列数量
        self.n_dim = self.M

#Column(coeffs,constrs)相应系数与约束
    def update_contrs(self, column_coeff):
        self.column = gp.Column(column_coeff, self.model.getConstrs())
        self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, obj=1, name='x' + str(self.n_dim), column=self.column)
        self.n_dim += 1
        self.n_col += 1

#ObjVal当前解的目标值
    def print_status(self):
        print("master objective value: {}".format(self.model.ObjVal))

#把变量属性改为整数
    def to_int(self):
        for x in self.model.getVars():
            x.setAttr("VType", GRB.INTEGER)

#把主问题写成一个文件名为model.lp
    def write(self):
        self.model.write("model.lp")


class SubProblem:   #子问题
    def __init__(self, lengths, W) -> None:
        self.lengths, self.M, self.W = lengths, len(lengths), W

#对偶变量y(i),对应每种长度木材(i)的数量
    def create_model(self):
        self.model = gp.Model("sub model")
        self.y = self.model.addVars(self.M, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='y')
        self.model.addConstr((gp.quicksum(self.lengths[i] * self.y[i] for i in range(self.M)) <= self.W))

#设置目标函数:max(判别数），已知判别数等于对偶数*变量-1，因此上式等价于max(Pi*yi)
    def set_objective(self, pi):
        self.model.setObjective(gp.quicksum(pi[i] * self.y[i] for i in range(self.M)), sense=GRB.MAXIMIZE)

    def solve(self, flag=0):
        self.model.Params.OutputFlag = flag
        self.model.optimize()

#返回每种木材各切多少，对应一个方案，也就是添加列(x)
    def get_solution(self):
        return [self.model.getVars()[i].x for i in range(self.M)]

    def get_reduced_cost(self):
        return self.model.ObjVal

    def write(self):
        self.model.write("sub_model.lp")


W = 20  # width of large roll
lengths = [3, 7, 9, 16]
demands = [25, 30, 14, 8]
M = len(lengths)  # number of items
N = sum(demands)  # number of available rolls

#Kantorovich模型，针对cutting_stocks直接的建模法
#model = gp.Model("cutting stock")
#y = model.addVars(N, vtype=GRB.BINARY, name='y')
#x = model.addVars(N, M, vtype=GRB.INTEGER, name='x')
#model.addConstrs((gp.quicksum(x[i, j] for i in range(N)) >= demands[j]) for j in range(M))
#model.addConstrs((gp.quicksum(lengths[j] * x[i, j] for j in range(M)) <= W * y[i] for i in range(N)))
#model.setObjective(gp.quicksum(y[i] for i in range(N)))

#model.optimize()

# x_j: number of times patter j is used
# a_ij: number of times item i is cut in patter j

#列生成算法
MAX_ITER_TIMES = 10

cutting_stock = Master(lengths, demands, W)
cutting_stock.create_model()
sub_prob = SubProblem(lengths, W)
sub_prob.create_model()

for k in range(MAX_ITER_TIMES):
    cutting_stock.solve()
    pi = cutting_stock.get_dual_vars()
    cutting_stock.write()

    sub_prob.set_objective(pi)
    sub_prob.solve()
    y = sub_prob.get_solution()
    reduced_cost = sub_prob.get_reduced_cost() #reduced_cost=判别数+1
    sub_prob.write()
    cutting_stock.update_contrs(column_coeff=y)
    if reduced_cost <= 1:  #即当判别数小于0，则达到最优
        break

cutting_stock.to_int()
cutting_stock.solve(flag=1)  #取整
#结果可通过lp文件看出