import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
import sys
import time
from mpi4py import MPI
import collections
import itertools
import seaborn as sns

class BCMP_Simulation:
    
    def __init__(self, N, R, K, U, mu, m, type_list, p, theoretical, default, sim_time, rank, size, comm):
        self.N = N #網内の拠点数(プログラム内部で指定する場合は1少なくしている)
        self.R = R #クラス数
        self.K = K #網内の客数 K = [K1, K2]のようなリスト。トータルはsum(K)
        self.U = U #最大窓口数
        self.mu = mu #サービス率 FCFSはクラス別で変えられない
        self.type_list = type_list #Type1(FCFS),Type2プロセッサシェアリング（Processor Sharing: PS）,Type3無限サーバ（Infinite Server: IS）,Type4後着順割込継続型（LCFS-PR)
        self.p = p
        self.m = m.values #窓口配列
        self.event = [[] for i in range(self.N)] #各拠点で発生したイベント(arrival, departure)を格納
        self.eventclass = [[] for i in range(self.N)] #各拠点でイベント発生時の客クラス番号
        self.eventqueue = [[] for i in range(N)] #各拠点でイベント発生時のqueueの長さ
        self.eventtime = [[] for i in range(N)] #各拠点でイベントが発生した時の時刻
        self.timerate = np.zeros((self.N, sum(self.K)+1))#拠点での人数分布(0~K人の分布)、0人の場合も入る
        self.timerateclass = np.zeros((self.N, self.R, sum(self.K)+1))#拠点での人数分布(0~K人の分布)、0人の場合も入る、クラス別
        self.time = sim_time #シミュレーション時間
        self.theoretical = theoretical
        self.default = np.floor(default.values)
        self.sum_defaultclass = np.sum(self.default, axis=0) #切り捨て後の各クラスの合計
        if rank == 0:
            #平均算出用
            self.sum_L = np.zeros(self.N) #平均系内人数(結果の和)
            self.sum_Lc = np.zeros((self.N, self.R)) # #平均系内人数(結果の和)(クラス別)
            self.sum_Q = np.zeros(self.N) #平均待ち人数(結果の和)
            self.sum_Qc = np.zeros((self.N, self.R)) #平均待ち人数(結果の和)(クラス別)
        #print(self.p.iloc[0,1])
        self.rank = rank
        self.size = size
        self.comm = comm
        self.process_text = './process/process_N'+str(self.N)+'_R'+str(self.R)+'_K'+str(sum(self.K))+'_U'+str(self.U)+'_Time'+str(self.time)+'.txt'
        self.start = time.time()
        
        
    def getSimulation(self):
        queue = np.zeros(self.N) #各拠点のサービス中を含むqueueの長さ(クラスをまとめたもの)
        queueclass = np.zeros((self.N, self.R)) #各拠点のサービス中を含むqueueの長さ(クラス別)
        classorder = [[] for i in range(self.N)] #拠点に並んでいる順のクラス番号
        window = np.full((self.N, self.U), self.R) #サービス中の客クラス(serviceに対応)(self.Rは空状態)
        service = np.zeros((self.N, self.U)) #サービス中の客の残りサービス時間
        total_length = np.zeros(self.N) #各拠点の延べ系内人数(クラスをまとめたもの)
        total_lengthclass = np.zeros((self.N, self.R)) #各拠点の延べ人数(クラス別)
        total_waiting = np.zeros(self.N) #延べ待ち人数(クラスをまとめたもの)
        total_waitingclass = np.zeros((self.N, self.R))#延べ待ち人数(クラス別)
        L = np.zeros(self.N) #平均系内人数(結果)
        Lc = np.zeros((self.N, self.R)) #平均系内人数(結果)(クラス別)
        Q = np.zeros(self.N) #平均待ち人数(結果)
        Qc = np.zeros((self.N, self.R)) #平均待ち人数(結果)(クラス別)
        rmse = [] #100単位時間でのrmseの値を格納
        rmse_time = [] #rmseを登録した時間
        regist_time = 50 #rmseの登録時刻
        regist_span = 50 #50単位で登録
        
        variation = [[] for i in range(self.N)] #系内人数の変化率
        length = [[] for i in range(self.N)] #系内人数の変化率
        L_list = [[] for i in range(self.N)] #各ノードの平均系内人数
        label_list = [] #boxplotのラベルリスト
        for n in range(1, self.N+1):
            label_list.append(n)
        
        elapse = 0
        #Step1 開始時の客の分配 (開始時のノードは拠点番号0)
        #今回は窓口数1の理論値を初期値として用いる
        for r in range(self.R):
            for n in range(self.N):
                if int(self.default[n][r]) > 0:
                    for k in range(int(self.default[n][r])):
                        self.event[n].append("arrival")
                        self.eventclass[n].append(r) #到着客のクラス番号
                        self.eventqueue[n].append(queue[n])#イベントが発生した時のqueueの長さ(到着客は含まない)
                        self.eventtime[n].append(elapse) #(移動時間0)
                        queue[n] +=1 #最初はノード0にn人いるとする
                        queueclass[n][r] += 1 #拠点0にクラス別人数を追加
                        classorder[n].append(r)#拠点0にクラス番号を追加
                        if queue[n] <= self.m[n][0]:#窓口数よりqueueが小さいとき
                            window[n][int(queue[n] - 1)] = r
                            service[n][int(queue[n] - 1)] = self.getExponential(self.mu[n]) #窓口客のサービス時間設定

            if self.K[r] > int(self.sum_defaultclass[r]): #足りない客は推移確率にしたがって配置
                pr = np.zeros((self.N, self.N))#客クラスの推移確率行列を抜き出す
                for i in range(self.N * r, self.N * (r + 1)):
                    for j in range(self.N * r, self.N * (r + 1)):
                        pr[i - self.N * r, j - self.N * r] = self.p.iloc[i,j]

                initial_node = 0
                for j in range(self.K[r] - int(self.sum_defaultclass[r])): #客の配置
                    rand = random.random()
                    sum_rand = 0
                    initial_index = -1
                    for i in range(len(pr)): #11   
                        sum_rand += pr[initial_node][i]
                        if rand < sum_rand:
                            initial_index = i
                            break
                    if initial_index == -1: #これは確率が1になってしまったとき用
                        initial_index = len(pr) -1 #一番最後のノードに移動することにする

                    initial_node = initial_index
                    self.event[initial_node].append("arrival")
                    self.eventclass[initial_node].append(r) #到着客のクラス番号
                    self.eventqueue[initial_node].append(queue[initial_node])#イベントが発生した時のqueueの長さ(到着客は含まない)
                    self.eventtime[initial_node].append(elapse) #(移動時間0)
                    queue[initial_node] +=1 #最初はノード0にn人いるとする
                    queueclass[initial_node][r] += 1 #拠点0にクラス別人数を追加
                    classorder[initial_node].append(r)#拠点0にクラス番号を追加
                    if queue[initial_node] <= self.m[initial_node][0]:#窓口数よりqueueが小さいとき
                        window[initial_node][int(queue[initial_node] - 1)] = r
                        service[initial_node][int(queue[initial_node] - 1)] = self.getExponential(self.mu[initial_node]) #窓口客のサービス時間設定
        
        
        #print('Simulation Start')
        #Step2 シミュレーション開始
        while elapse < self.time:
            #print('経過時間 : {0} / {1}'.format(elapse, self.time))
            mini_service = 100000#最小のサービス時間
            mini_index = -1 #最小のサービス時間をもつノード
            window_index = -1 #最小のサービス時間の窓口
           
            #print('Step2.1 次に退去が起こる拠点を検索')
            #Step2.1 次に退去が起こる拠点を検索
            for i in range(self.N):#待ち人数がいる中で最小のサービス時間を持つノードの窓口を算出
                if queue[i] > 0:
                    c = 0
                    for j in range(self.m[i][0]):
                        if window[i][j] < self.R:
                            c += 1
                            if mini_service > service[i][j]:
                                mini_service = service[i][j]
                                mini_index = i
                                window_index = j
                                classorder_index = c-1
            departure_class = classorder[mini_index].pop(classorder_index) #退去する客のクラスを取り出す

    
            #Step2.2 イベント拠点確定後、全ノードの情報更新(サービス時間、延べ人数)
            for i in range(self.N):#ノードiから退去(全拠点で更新)
                total_length[i] += queue[i] * mini_service #ノードでの延べ系内人数
                for r in range(self.R): #クラス別延べ人数更新
                    total_lengthclass[i,r] += queueclass[i,r] * mini_service
                if queue[i] > 0: #系内人数がいる場合(サービス中の客がいるとき)
                    for j in range(self.m[i][0]):
                        if service[i][j] > 0:
                            service[i][j] -= mini_service #サービス時間を減らす
                    if queue[i] > self.m[i][0]:
                        total_waiting[i] += ( queue[i] - self.m[i][0] ) * mini_service #ノードでの延べ待ち人数(queueの長さが窓口数以上か以下かで変わる)
                    else:
                        total_waiting[i] += 0 * mini_service
                    for r in range(R):
                        if queueclass[i,r] > 0: #クラス別延べ待ち人数の更新(windowの各クラスの数をカウント -> カウント分queueclassから引く)
                            c = np.count_nonzero(window[i]==r)
                            total_waitingclass[i,r] += ( queueclass[i,r] - c ) * mini_service
                            #print('クラス別延べ待ち人数window[{0}] : {1} c={2}'.format(i, window[i], c)) 
                elif queue[i] == 0: #いらないかも
                    total_waiting[i] += queue[i] * mini_service
                self.timerate[i, int(queue[i])] += mini_service #人数分布の時間帯を更新
                for r in range(R):
                    self.timerateclass[i, r, int(queueclass[i,r])] += mini_service #人数分布の時間帯を更新
            
        
            #Step2.3 退去を反映
            self.event[mini_index].append("departure") #退去を登録
            self.eventclass[mini_index].append(departure_class)
            self.eventqueue[mini_index].append(queue[mini_index]) #イベント時の系内人数を登録
            #self.eventqueueclass[mini_index, departure_class].append(queueclass[mini_index, departure_class]) #イベント時の系内人数を登録
            queue[mini_index] -= 1 #ノードの系内人数を減らす
            queueclass[mini_index, departure_class] -= 1 #ノードの系内人数を減らす(クラス別)
            elapse += mini_service
            self.eventtime[mini_index].append(elapse) #経過時間の登録はイベント後
            window[mini_index][window_index] = self.R

            if queue[mini_index] > 0: #窓口がすべて埋まっている -> １つ空いた
                for i in range(self.m[mini_index][0]):
                    if service[mini_index][i] == 0 and window[mini_index][i] == self.R and queue[mini_index] >= self.m[mini_index][0]: #もう一つ条件を追加
                        window[mini_index][i] = classorder[mini_index][int(self.m[mini_index][0] - 1)]
                        service[mini_index][i] = self.getExponential(self.mu[mini_index])#退去後まだ待ち人数がある場合、サービス時間設定
                        break

            
            #Step2.4 退去客の行き先決定
            #推移確率行列が N*R × N*Rになっている。departure_class = 0の時は最初のN×N (0~N-1の要素)を見ればいい
            #departure_class = 1の時は (N~2N-1の要素)、departure_class = 2の時は (2N~3N-1の要素)
            #departure_class = rの時は (N*r~N*(r+1)-1)を見ればいい
            rand = random.random()
            sum_rand = 0
            destination_index = -1
            pr = np.zeros((self.N, self.N))#今回退去する客クラスの推移確率行列を抜き出す
            for i in range(self.N * departure_class, self.N * (departure_class + 1)):
                for j in range(self.N * departure_class, self.N * (departure_class + 1)):
                    pr[i - self.N * departure_class, j - self.N * departure_class] = self.p.iloc[i,j]
            
            for i in range(len(pr)): #11   
                sum_rand += pr[mini_index][i]
                if rand < sum_rand:
                    destination_index = i
                    break
            if destination_index == -1: #これは確率が1になってしまったとき用
                destination_index = len(pr) -1 #一番最後のノードに移動することにする
            self.event[destination_index].append("arrival") #イベント登録
            self.eventclass[destination_index].append(departure_class) #移動する客クラス番号登録
            self.eventqueue[destination_index].append(queue[destination_index])
            self.eventtime[destination_index].append(elapse) #(移動時間0)
            queue[destination_index] += 1 #推移先の待ち行列に並ぶ
            queueclass[destination_index][departure_class] += 1 #推移先の待ち行列(クラス別)に登録 
            classorder[destination_index].append(departure_class)
            #推移先で待っている客がいなければサービス時間設定(即時サービス)
            if queue[destination_index] <= self.m[destination_index][0]:
                for i in range(self.m[destination_index][0]):
                    if service[destination_index][i] == 0 and window[destination_index][i] == self.R:
                        window[destination_index][i] = classorder[destination_index][-1]
                        service[destination_index][i] = self.getExponential(self.mu[destination_index]) #サービス時間設定
                        break
            
           
            #Step2.5 RMSEの計算
            if elapse > regist_time:
                rmse_sum = 0
                theoretical_value = self.theoretical.values
                theoretical_row = np.sum(self.theoretical.values, axis=1)
                l = total_length / elapse #今までの時刻での平均系内人数
                lc = total_lengthclass / elapse #今までの時刻での平均系内人数(クラス別)
                for n in range(self.N):
                    for r in range(self.R):
                        rmse_sum += (theoretical_value[n,r] - lc[n,r])**2
                rmse_sum /= self.N * self.R
                rmse_value = math.sqrt(rmse_sum)
                rmse.append(rmse_value)
                rmse_time.append(regist_time)
                regist_time += regist_span
                print('Rank = {0}, Calculation Time = {1}'.format(self.rank, time.time() - self.start))
                print('Elapse = {0}, RMSE = {1}'.format(elapse, rmse_value))
                print('Elapse = {0}, Lc = {1}'.format(elapse, lc))
                if self.rank == 0:
                    with open(self.process_text, 'a') as f:
                        print('Rank = {0}, Calculation Time = {1}'.format(self.rank, time.time() - self.start), file=f)
                        print('Elapse = {0}, RMSE = {1}'.format(elapse, rmse_value), file=f)
                        print('Elapse = {0}, Lc = {1}'.format(elapse, lc), file=f)

                #時間経過による人数変化
                for n in range(self.N):
                    #l_variation = l[n] - theoretical_row[n]
                    #variation[n].append(l_variation)
                    len_variation = queue[n] - theoretical_row[n]
                    variation[n].append(len_variation)
                    length[n].append(queue[n])
                    L_list[n].append(l[n])

                #5000時間でboxplot(平均系内人数)
                if 5000 <= regist_time and regist_time < (5000 + regist_span):
                    plt.figure(figsize=(12,5))
                    plt.boxplot(L_list, labels=label_list)
                    plt.savefig('./plot/L_box_Time5000_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+'.png', format='png', dpi=300,  bbox_inches='tight')
                    plt.clf()
                    plt.close()
        


        L = total_length / self.time
        Lc = total_lengthclass / self.time
        Q = total_waiting / self.time
        Qc = total_waitingclass / self.time
        
        print('Rank = {0}, Calculation Time = {1}'.format(self.rank, time.time() - self.start))
        print('平均系内人数L : {0}'.format(L))
        print('平均系内人数(クラス別)Lc : {0}'.format(Lc))
        print('平均待ち人数Q : {0}'.format(Q))
        print('平均待ち人数(クラス別)Qc : {0}'.format(Qc))
        if self.rank == 0:
            with open(self.process_text, 'a') as f:
                print('Rank = {0}, Calculation Time = {1}'.format(self.rank, time.time() - self.start), file=f)
                print('平均系内人数L : {0}'.format(L), file=f)
                print('平均系内人数(クラス別)Lc : {0}'.format(Lc), file=f)
                print('平均待ち人数Q : {0}'.format(Q), file=f)
                print('平均待ち人数(クラス別)Qc : {0}'.format(Qc), file=f)
                   
        pd.DataFrame(L).to_csv('./csv/L(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')
        pd.DataFrame(Lc).to_csv('./csv/Lc(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')
        pd.DataFrame(Q).to_csv('./csv/Q(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')
        pd.DataFrame(Qc).to_csv('./csv/Qc(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')
        rmse_index = {'time': rmse_time, 'RMSE': rmse}
        df_rmse = pd.DataFrame(rmse_index)
        df_rmse.to_csv('./csv/RMSE(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')


        plt.figure(figsize=(12,5))
        colorlist = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'mediumaquamarine']
        #人の推移
        for n in range(self.N):
            if 0 <= n and n < 11:
                plt.plot(rmse_time, length[n], '-', lw=0.5, color=colorlist[n], label='node'+str(n))
            elif 11 <= n and n < 22:
                plt.plot(rmse_time, length[n], '--', lw=0.5, color=colorlist[n-11], label='node'+str(n))
            else:
                plt.plot(rmse_time, length[n], ':', lw=0.5, color=colorlist[n-22], label='node'+str(n))
        plt.legend(fontsize='xx-small', ncol=3, bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.1)
        plt.grid(which='major', axis='y', color='black', alpha=0.5, linestyle='-', linewidth=0.5)
        plt.savefig('./plot/length_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+'.png', format='png', dpi=300,  bbox_inches='tight')
        plt.clf()

        #boxplot(平均系内人数)
        plt.boxplot(L_list, labels=label_list)
        plt.savefig('./plot/box_L_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+'.png', format='png', dpi=300,  bbox_inches='tight')
        plt.clf()
        plt.close()

        #相関係数行列
        plt.figure(figsize=(9,6))
        #L_list(平均系内人数)とlength(系内人数)の転置行列
        L_list = np.array(L_list)
        L_list_T = L_list.T
        df = pd.DataFrame(L_list_T, columns=label_list)#データフレーム化
        df_corr = df.corr() #相関係数行列
        #df.to_csv('./csv/L_variation(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')
        df_corr.to_csv('./csv/L_corr(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+').csv')
        sns.heatmap(df_corr, cmap="bwr") #ヒートマップ
        sns.set(font_scale=0.7) # font size 0.5
        plt.savefig('./plot/heatmap_L_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+'.png', format='png', dpi=300,  bbox_inches='tight')
        plt.clf()

        #各rankのRMSE最終値を取得
        if self.rank == 0:
            last_rmse = []
            last_rmse.append(rmse[-1])

            for i in range(1, self.size):
                rmse_rank = self.comm.recv(source=i, tag=0)
                last_rmse.append(rmse_rank[-1])

        else:
            self.comm.send(rmse, dest=0, tag=0)
        self.comm.barrier() #プロセス同期

        #RMSEが最小と最大のものを除いて、平均系内人数の平均を算出
        if self.rank == 0:
            plt.figure(figsize=(12,5))
            #RMSEが最大と最小のrank取得
            max_index = last_rmse.index(max(last_rmse))
            min_index = last_rmse.index(min(last_rmse))

            #plot及び平均系内人数の平均の算出
            for i in range(0, self.size):
                if i > 0:
                    L_rank = self.comm.recv(source=i, tag=1)
                    Lc_rank = self.comm.recv(source=i, tag=2)
                    Q_rank = self.comm.recv(source=i, tag=3)
                    Qc_rank = self.comm.recv(source=i, tag=4)
                    rmse_rank = self.comm.recv(source=i, tag=10)
                    time_rank = self.comm.recv(source=i, tag=11)

                    if i == max_index or i == min_index:
                        plt.plot(time_rank, rmse_rank, linestyle = 'dotted', color = 'black', alpha = 0.2) #平均に含まれない
                    else:
                        self.sum_L += L_rank
                        self.sum_Lc += Lc_rank
                        self.sum_Q += Q_rank
                        self.sum_Qc += Qc_rank
                        plt.plot(time_rank, rmse_rank)
                else:
                    if i == max_index or i == min_index:
                        plt.plot(rmse_time, rmse, linestyle = 'dotted', color = 'black', alpha = 0.2) #平均に含まれない
                    else:
                        self.sum_L += L
                        self.sum_Lc += Lc
                        self.sum_Q += Q
                        self.sum_Qc += Qc
                        plt.plot(rmse_time, rmse)

            #全体平均の算出
            avg_L = self.sum_L / (self.size - 2)
            avg_Lc = self.sum_Lc / (self.size - 2)
            avg_Q = self.sum_Q / (self.size - 2)
            avg_Qc = self.sum_Qc / (self.size - 2)
            print('----- シミュレーション結果 -----')
            print('平均系内人数avg_L : {0}'.format(avg_L))
            print('平均系内人数(クラス別)avg_Lc : {0}'.format(avg_Lc))
            print('平均待ち人数avg_Q : {0}'.format(avg_Q))
            print('平均待ち人数(クラス別)avg_Qc : {0}'.format(avg_Qc))

            pd.DataFrame(avg_L).to_csv('./csv/avg_L(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Size_'+str(self.size)+').csv')
            pd.DataFrame(avg_Lc).to_csv('./csv/avg_Lc(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Size_'+str(self.size)+').csv')
            pd.DataFrame(avg_Q).to_csv('./csv/avg_Q(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Size_'+str(self.size)+').csv')
            pd.DataFrame(avg_Qc).to_csv('./csv/avg_Qc(N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Size_'+str(self.size)+').csv')
            plt.savefig('./plot/N_'+str(self.N)+',R_'+str(self.R)+',K_'+str(self.K)+',U_'+str(self.U)+',Time_'+str(self.time)+',Rank_'+str(self.rank)+'.png', format='png', dpi=300)
            plt.clf()
            plt.close()
        
        else:
            self.comm.send(L, dest=0, tag=1)
            self.comm.send(Lc, dest=0, tag=2)
            self.comm.send(Q, dest=0, tag=3)
            self.comm.send(Qc, dest=0, tag=4)
            self.comm.send(rmse, dest=0, tag=10)
            self.comm.send(rmse_time, dest=0, tag=11)

        
    def getExponential(self, param):
        return - math.log(1 - random.random()) / param
    
    
    
if __name__ == '__main__':
   
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #推移確率行列に合わせる
    N = int(sys.argv[1])
    R = int(sys.argv[2])
    K_total = int(sys.argv[3])
    U = int(sys.argv[4])
    sim_time = int(sys.argv[5])
    p_file = sys.argv[6]
    theoretical_file = sys.argv[7]
    m_file = sys.argv[8]
    default_file = sys.argv[9] #系内人数の初期値(今回は窓口数1の理論値を使用)
    #N = 33 #33
    #R = 2
    #K_total = 500
    #U = 4
    K = [(K_total + i) // R for i in range(R)] #クラス人数を自動的に配分する
    mu = np.full(N, 1) #サービス率を同じ値で生成(サービス率は調整が必要)
    m = pd.read_csv(m_file, header=None, dtype=int)
    type_list = np.full(N, 1) #サービスタイプはFCFS
    p = pd.read_csv(p_file, header=None)
    theoretical = pd.read_csv(theoretical_file, header=None)
    default = pd.read_csv(default_file, header=None)
    bcmp = BCMP_Simulation(N, R, K, U, mu, m, type_list, p, theoretical, default, sim_time, rank, size, comm) 
    start = time.time()
    bcmp.getSimulation()
    elapsed_time = time.time() - start
    print ("rank : {1}, calclation_time:{0}".format(elapsed_time, rank) + "[sec]")
    
   
    #mpiexec -n 8 python3 BCMP_Simulation_v2.py 33(ノード数) 2(クラス数) 500(網内人数) 2(最大窓口数) 100000(シミュレーション時間) P_std_33_2_500_76_50.csv(推移確率行列) ga_L_std_33_2_500_76_50.csv(理論値) ga_Node_std_33_2_500_76_50.csv(窓口数) ga_L_std_33_2_500_76_50_1.csv(初期値) > result_33_2_500_2_100000.txt
    #mpiexec -n 8 python3 BCMP_Simulation_v2.py 33 2 500 2 100000 P_std_33_2_500_76_50.csv ga_L_std_33_2_500_76_50.csv ga_Node_std_33_2_500_76_50.csv ga_L_std_33_2_500_76_50_1.csv > result_33_2_500_2_100000.txt
    #mpiexec -n 6 python BCMP_Simulation_v2.py 33 2 500 2 500 P_std_33_2_500_76_50.csv ga_L_std_33_2_500_76_50.csv ga_Node_std_33_2_500_76_50.csv ga_L_std_33_2_500_76_50_1.csv > result_33_2_500_2_500_1.txt
    