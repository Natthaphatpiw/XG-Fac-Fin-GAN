# รายชื่อหุ้นที่ต้องการโหลดข้อมูล
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import time
from memory_profiler import memory_usage
from functools import partial
import json
import joblib

# ---------------------- Specify the stocks that st -------------------------------------------------------------

st = "KBANK"


# stocks = ["KBANK", "MINT", "BBL", "LH", "CPALL"]
stocks = [st]

data_stocks = {}
df_result = pd.DataFrame(index=["RMSE", "MAE", "PnL", "Sharpe Ratio", "Sharpe Ratio Daily","Processing Time","Memory Usage"])

# print(df_result)



# ---------------------- XGBoost -----------------------------------------------------------------------------------
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import time
import json
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from memory_profiler import memory_usage
from functools import partial

def load_data(stock):
    data = pd.read_csv(f"{stock}_final.csv")
    data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y', errors='coerce')
    data = data.sort_values(by='date').drop(columns='date')
    return data

def preprocess_data(data):
    data['log'] = np.log(data['cp'])
    data['ret'] = data['log'].diff()
    data['nret'] = data['ret'].shift(-1) 
    data = data.drop(columns=['cp', 'log', 'ret']).dropna().reset_index(drop=True)
    return data

def normalize_data(data):
    X = data.iloc[:, :-1]
    y = data['nret']
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), y, scaler


def split_data(X, y, test_size=0.05, valid_size=0.2):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=valid_size/(test_size+valid_size), shuffle=False)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


# ---------------------- Set parameters and train -------------------------------------------------------------

def train_xgboost(X_train, y_train, X_valid, y_valid):
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.00005,
        'max_depth': 10,
        'n_estimators': 500,
        'gamma': 0.001,
        'random_state': 42
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=False)
    return model



# ----------------------Plot feature importance  -------------------------------------------------------------

def plot_feature_importance(model, stock):
    plt.figure(figsize=(12, 8))  # ขยายขนาดรูป
    
    # ดึง Feature Importance
    importance = model.get_booster().get_score(importance_type='weight')
    importance = {k: v / max(importance.values()) for k, v in importance.items()}  # Normalize 0-1
    
    # สร้าง DataFrame เพื่อจัดเรียงข้อมูล
    importance_df = pd.DataFrame(list(importance.items()), columns=["Feature", "Importance"])
    importance_df = importance_df.sort_values(by="Importance", ascending=True)

    # วาดกราฟ
    plt.barh(importance_df["Feature"], importance_df["Importance"], color="royalblue", height=0.5)
    plt.xlabel("Importance (0-1)", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(f"Feature Importance : {stock}", fontsize=14)
    plt.xlim(0, 1)  # กำหนดให้แกน X อยู่ในช่วง 0-1
    plt.grid(False)
    
    plt.savefig(f"XGBoost_{stock}_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()

    # คัดเลือก 5 Feature ที่มีความสำคัญต่ำสุด
    least_important_features = importance_df["Feature"].iloc[:5].tolist()
    
    # โหลด DataFrame และลบ Features ที่ไม่สำคัญออก
    data_stocks = pd.read_csv(f"{stock}_final.csv")
    data_stocks_filtered = data_stocks.drop(columns=least_important_features, errors='ignore')
    
    # จำนวน Feature ที่เหลืออยู่
    fea = data_stocks_filtered.shape[1] - 1  # ลบ 1 เพราะต้องไม่รวมคอลัมน์ target (ถ้ามี)

    return data_stocks_filtered, fea


#
# stocks = ["BBL"]
# stocks = ["KBANK", "MINT", "BBL", "LH", "CPALL"]
# df_result = pd.DataFrame(index=["RMSE", "MAE", "Processing Time", "Memory Usage"])

for stock in stocks:
    data = load_data(stock)
    data = preprocess_data(data)
    X, y, scaler = normalize_data(data)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(X, y)
    
    start_time = time.time()
    model = train_xgboost(X_train, y_train, X_valid, y_valid)
    end_time = time.time()
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    mae = np.mean(np.abs(y_pred - y_test))
    
    data_stocks_filtered, fea = plot_feature_importance(model, stock)
    
    processing_time = end_time - start_time
    memory_usage_train = memory_usage(partial(train_xgboost, X_train, y_train, X_valid, y_valid))
    memory_usage_value = max(memory_usage_train) - min(memory_usage_train)
    
    # df_result[stock] = [rmse, mae, processing_time, memory_usage_value]
    
    # joblib.dump(model, f"xgboost_{stock}_model.pkl")

# df_result.to_csv("XGBoost_All_result.csv", index=True)




# ---------------------- Factor-Fin-GAN -----------------------------------------------------------------------------------


# วนลูปโหลดข้อมูลของแต่ละหุ้น
for stock in stocks:
    filename = f"{stock}_final.csv"  # ตั้งชื่อไฟล์ให้อัตโนมัติ เช่น "KBANK_final.csv"
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import random
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy.random as rnd
    from sklearn.preprocessing import MinMaxScaler
    import time
    from memory_profiler import memory_usage

    # Data preprocessing --------------------------------------------------------------------------------------------------------

    print("Data preprocessing ----------------------------------------------------------------------")

    tickers = stock
    data_stocks = data_stocks_filtered
    # แปลงวันที่ (ระบุ format ให้ตรงกับข้อมูล)
    data_stocks['date'] = pd.to_datetime(data_stocks['date'], format='%d/%m/%Y', errors='coerce')

    ### การตั้งค่าพื้นฐาน
    h = 1
    l = 10                  # lag
    pred = 1
    fea = fea-1
    ngpu = 1

    ### เส้นทางข้อมูล
    dataloc = "./"
    loc = "./"
    modelsloc = loc + "./"
    plotsloc = loc + "./"
    resultsloc = loc + "./"

    tanh_coeff = 10          # ลดความรุนแรงของฟังก์ชัน tanh
    z_dim = 32               # latent space
    hid_d = 32               # hidden units in discriminator
    hid_g = 32               # hidden units in generator
    checkpoint_epoch = 5    
    batch_size = 16          
    n_epochs = 400            # เพิ่มรอบการเทรน
    tr = 0.5                 # train set
    vl = 0.1                 # validation set

    lrd = 0.00005           # learning rate discriminator
    lrg = 0.00005           # learning rate generator
    diter =1
    plot=False
    ticker = stock
    f_name = modelsloc+ ticker + "-Fin-GAN-"+str(n_epochs)+"-epochs-"+str(lrd)+"-lrd-"+str(lrg)+"-lrg-"
    f_name1 = ticker + "-Fin-GAN-"+str(n_epochs)+"-epochs-"+str(lrd)+"-lrd-"+str(lrg)+"-lrg"
    PnL_test = [False] * 10

    print("SR MSE Val")
    losstype = "SR MSE Val"

    resultsname = "results.csv"
    plt.rcParams['figure.figsize'] = [15.75, 9.385]

    datastart = {'lrd': [], 'lrg': [], 'epochs': [], 'SR_val': []}
    results_df = pd.DataFrame(data=datastart)

    def rawreturns(df,dataloc, stock):

        # s_df = pd.read_csv(dataloc+stock+".csv")
        s_df = df
        dates_dt = pd.to_datetime(s_df['date'])
        s_logclose = np.log(s_df['cp'])
        s_ret = np.diff(s_logclose)  # คำนวณ log returns
        dates_dt = pd.to_datetime(s_df['date'])

        return s_ret, dates_dt


    print("rawreturns...")
    excess_returns, dates_dt = rawreturns(data_stocks,dataloc, stock)
    print("end rawreturns")

    print("Len excess_returns =",len(excess_returns))

    # Set the date to datetime data
    dataset =  data_stocks
    datetime_series = pd.to_datetime(dataset['date'])
    datetime_index = pd.DatetimeIndex(datetime_series.values)
    dataset = dataset.set_index(datetime_index)
    dataset = dataset.sort_values(by='date')
    dataset = dataset.drop(columns='date')

    stock_df = dataset
    stock_df['log'] = np.log(stock_df['cp'])
    stock_df['ret'] = stock_df['log'].diff()
    stock_df['nret'] = stock_df['ret'].shift(-1)
    drop_cols = ['cp','log','ret']
    stock_df = stock_df.drop(columns=drop_cols)
    stock_df = stock_df.dropna().reset_index(drop=True)
    X_value = pd.DataFrame(stock_df.iloc[:,:-1])

    # Normalized the data
    X_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaler.fit(X_value)
    X_scale_dataset = X_scaler.fit_transform(X_value)

    X_scale_dataset = pd.DataFrame(X_scale_dataset)
    dataset = X_scale_dataset
    # drop_cols = [9,11,6,13,18]
    # dataset = dataset.drop(columns=drop_cols)
    plotcheck = False

    def split_train_val_testraw(df,stock, dataloc, tr = 0.8, vl = 0.1, h = 1, l = 10,fea = fea, pred = 1, plotcheck=False):

        excess_returns, dates_dt = rawreturns(data_stocks,dataloc, stock)
        N = len(excess_returns)
        print("Len N =",N)
        N_tr = int(tr*N)
        print("Len N_tr =",N_tr)
        N_vl = int(vl*N)
        print("Len N_vl =",N_vl)
        N_tst = N - N_tr - N_vl
        print("Len N_tst =",N_tst)
        train_sr = excess_returns[0:N_tr]

        train_fea = pd.DataFrame(df.iloc[0:N_tr,:])

        val_sr = excess_returns[N_tr:N_tr+N_vl]

        val_fea = pd.DataFrame(df.iloc[N_tr:N_tr+N_vl,:])

        test_sr = excess_returns[N_tr+N_vl:]

        test_fea = pd.DataFrame(df.iloc[N_tr+N_vl:,:])

        n = int((N_tr-l-pred)/h)+1
        print("Len sample int((N_tr-l-pred)/h)+1 =",n)
        train_data = np.zeros(shape=(n,fea+l+pred))
        l_tot = 0
        for i in tqdm(range(n)):
            train_data[i, :fea] = train_fea.iloc[l_tot + l + pred - 1, :fea]
            train_data[i, fea:fea + l + pred] = train_sr[l_tot:l_tot + l + pred]
            l_tot = l_tot + h


        n = int((N_vl-l-pred)/h)+1
        val_data = np.zeros(shape=(n, fea + l + pred))  # (19 ฟีเจอร์ + 10 ค่า Return + 1 เป้าหมาย)

        l_tot = 0
        for i in tqdm(range(n)):
            val_data[i, :fea] = val_fea.iloc[l_tot + l + pred - 1, :fea]  # ใส่ 19 ฟีเจอร์
            val_data[i, fea:fea + l + pred] = val_sr[l_tot:l_tot + l + pred]  # ใส่ค่า Return
            l_tot = l_tot + h


        n = int((N_tst-l-pred)/h)+1
        test_data = np.zeros(shape=(n, fea + l + pred))  # (19 ฟีเจอร์ + 10 ค่า Return + 1 เป้าหมาย)

        l_tot = 0
        for i in tqdm(range(n)):
            test_data[i, :fea] = test_fea.iloc[l_tot + l + pred - 1, :fea]  # ใส่ 19 ฟีเจอร์
            test_data[i, fea:fea + l + pred] = test_sr[l_tot:l_tot + l + pred]  # ใส่ค่า Return
            l_tot = l_tot + h

        return train_data, val_data, test_data, dates_dt  # ✅ เพิ่ม return

    print("Start split_train_val_testraw ---------------------------------------------------------------------")
    train_data,val_data,test_data, dates_dt = split_train_val_testraw(dataset,stock, dataloc, tr, vl, h, l, fea, pred, plotcheck = False)
    print("End split_train_val_testraw ---------------------------------------------------------------------")

    datastart = {'lrd':[],'lrg':[],'epochs':[],'SR_val':[]}
    results_df = pd.DataFrame(data=datastart)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    data_tt = torch.from_numpy(train_data)
    train_data = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(test_data)
    test_data = data_tt.to(torch.float).to(device)
    data_tt = torch.from_numpy(val_data)
    validation_data = data_tt.to(torch.float).to(device)
    ntest = test_data.shape[0]

    condition_size = l
    target_size = pred
    feature_size = fea  # ขนาดของฟีเจอร์ที่เพิ่มเข้าไป

    # คำนวณ mean และ std จากเฉพาะส่วนของ Return และ Target Return
    ref_mean = torch.mean(train_data[0:batch_size, feature_size:])
    ref_std = torch.std(train_data[0:batch_size, feature_size:])

    # ปรับขนาด input ของ Discriminator ให้รองรับ Feature ด้วย
    discriminator_indim = feature_size + condition_size + target_size

    ##############################
    # Generator
    ##############################
    class Generator(nn.Module):
        def __init__(self, noise_dim, feature_size, cond_dim, hidden_dim, output_dim, mean, std):
            super(Generator, self).__init__()
            self.input_dim = noise_dim + feature_size + cond_dim
            self.feature_size = feature_size
            self.cond_dim = cond_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.noise_dim = noise_dim
            self.mean = mean
            self.std = std

            # LSTM รับ Input เป็น (features + conditions)
            self.lstm = nn.LSTM(input_size=feature_size + cond_dim, hidden_size=self.hidden_dim, num_layers=1, dropout=0)
            nn.init.xavier_normal_(self.lstm.weight_ih_l0)
            nn.init.xavier_normal_(self.lstm.weight_hh_l0)

            # Fully connected layers
            self.linear1 = nn.Linear(in_features=self.hidden_dim + self.noise_dim, out_features=self.hidden_dim + self.noise_dim)
            nn.init.xavier_normal_(self.linear1.weight)
            self.linear2 = nn.Linear(in_features=self.hidden_dim + self.noise_dim, out_features=output_dim)
            nn.init.xavier_normal_(self.linear2.weight)
            self.activation = nn.ReLU()

        def forward(self, noise, features, condition, h_0, c_0):
            # Normalize features & conditions
            condition = (condition - self.mean) / self.std
            input_lstm = torch.cat((features, condition), dim=-1)  # รวม features + conditions
            out, (h_n, c_n) = self.lstm(input_lstm, (h_0, c_0))
            out = combine_vectors(noise.to(torch.float), h_n.to(torch.float), dim=-1)
            out = self.linear1(out)
            out = self.activation(out)
            out = self.linear2(out)
            out = out * self.std + self.mean  # Denormalization
            return out

    ##############################
    # Discriminator
    ##############################
    class Discriminator(nn.Module):
        def __init__(self, in_dim, hidden_dim, mean, std):
            super(Discriminator, self).__init__()
            self.hidden_dim = hidden_dim
            self.mean = mean
            self.std = std

            # LSTM input size ต้องรองรับทั้ง features, conditions และ target
            self.lstm = nn.LSTM(input_size=in_dim, hidden_size=self.hidden_dim, num_layers=1, dropout=0)
            nn.init.xavier_normal_(self.lstm.weight_ih_l0)
            nn.init.xavier_normal_(self.lstm.weight_hh_l0)

            # Fully connected layer
            self.linear = nn.Linear(in_features=self.hidden_dim, out_features=1)
            nn.init.xavier_normal_(self.linear.weight)
            self.sigmoid = nn.Sigmoid()

        def forward(self, in_chan, h_0, c_0):
            x = (in_chan - self.mean) / self.std
            out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
            out = self.linear(h_n)
            out = self.sigmoid(out)
            return out

    ##############################
    # Function for Combining Vectors
    ##############################
    def combine_vectors(*args, dim=-1):
        """
        Function for combining multiple tensors dynamically
        """
        combined = torch.cat(args, dim=dim)  # ใช้ *args เพื่อรวมหลาย Tensor
        return combined.to(torch.float)

    ##############################
    # Profit & Loss Calculation
    ##############################
    def getPnL(stock, predicted, real, nsamp, plot=True):
        """
        PnL per trade given nsamp samples, predicted forecast, real data realisations
        in bpts
        """
        forecast = predicted
        sgn_fake = torch.sign(predicted)
        PnL_per_trade = sgn_fake * real   # คำนวณกำไรต่อการเทรด
        cumulative_PnL = torch.cumsum(PnL_per_trade, dim=0)  # กำไรสะสม
        std_pnl = torch.std(sgn_fake * real)

        PnL = torch.sum(sgn_fake * real)
        meanPnL = PnL / nsamp
        PnL = 10000 * PnL / nsamp  # Scaling

        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(cumulative_PnL.cpu().numpy(), label="Cumulative PnL", color='b')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
            plt.xlabel("Trade Number")
            plt.ylabel("Cumulative PnL")
            plt.title(f"Factor-Fin-GAN Backtest: Cumulative return of {stock}")
            plt.legend()
            plt.savefig(f"Factor-Fin-GAN_{stock}_cumulative_pnl.png", dpi=300, bbox_inches='tight')
            plt.show()
            # บันทึกภาพเป็นไฟล์ PNG


        return PnL, meanPnL, std_pnl,cumulative_PnL,sgn_fake,forecast

    gen = Generator(
        noise_dim=z_dim,
        feature_size=feature_size,  # ✅ เพิ่ม feature_size เข้าไป
        cond_dim=condition_size,
        hidden_dim=hid_g,
        output_dim=pred,
        mean=ref_mean,
        std=ref_std
    )
    gen.to(device)
    print("gen")
    print(gen)

    # ต้องแก้ไข input_size ของ Discriminator ให้รวมฟีเจอร์
    discriminator_indim = feature_size + condition_size + target_size  # ✅ แก้ไขให้รองรับ feature_size

    disc = Discriminator(
        in_dim=discriminator_indim,  # ✅ แก้ input size ให้รวมฟีเจอร์
        hidden_dim=hid_d,
        mean=ref_mean,
        std=ref_std
    )
    disc.to(device)
    print("dis")
    print(disc)

    # Optimizers (ไม่ต้องแก้ไข)
    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=lrg)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=lrd)
    criterion = nn.BCELoss()
    criterion = criterion.to(device)
    print("criterion")
    print(criterion)

    def get_gradient_norm(model):
        """
        คำนวณค่า Gradient Norm ของโมเดลที่กำหนด
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:  # เช็คว่ามี Gradient หรือไม่
                param_norm = p.grad.detach().data.norm(2)  # คำนวณ Norm ของพารามิเตอร์
                total_norm += param_norm.item() ** 2  # รวมค่า Gradient Norm
        return total_norm ** 0.5  # คำนวณค่า Norm สุดท้าย


    def GradientCheck( gen, disc, gen_opt, disc_opt, criterion, n_epochs, train_data, batch_size, hid_d, hid_g, z_dim, lr_d=0.0001, lr_g=0.0001, h=1, l=10, fea=19, pred=1, diter=1, tanh_coeff=100, device='cpu', plot=False):
        """
        Gradient norm check (รองรับการเพิ่ม Feature)
        """
        ntrain = train_data.shape[0]
        nbatches = ntrain // batch_size + 1

        BCE_norm = torch.empty(nbatches * n_epochs, device=device)
        PnL_norm = torch.empty(nbatches * n_epochs, device=device)
        MSE_norm = torch.empty(nbatches * n_epochs, device=device)
        SR_norm = torch.empty(nbatches * n_epochs, device=device)
        STD_norm = torch.empty(nbatches * n_epochs, device=device)

        gen.train()

        for epoch in tqdm(range(n_epochs)):
            perm = torch.randperm(ntrain)
            train_data = train_data[perm, :]

            for i in range(nbatches):
                curr_batch_size = batch_size
                if i == (nbatches - 1):
                    curr_batch_size = ntrain - i * batch_size

                # ✅ กำหนดค่า hidden state ของ LSTM
                h_0d = torch.zeros((1, curr_batch_size, hid_d), device=device, dtype=torch.float)
                c_0d = torch.zeros((1, curr_batch_size, hid_d), device=device, dtype=torch.float)
                h_0g = torch.zeros((1, curr_batch_size, hid_g), device=device, dtype=torch.float)
                c_0g = torch.zeros((1, curr_batch_size, hid_g), device=device, dtype=torch.float)

                # ✅ ดึง Feature (19 ตัว) + Condition (10 ตัว) ✅
                feature_data = train_data[(i * batch_size):(i * batch_size + curr_batch_size), 0:fea]
                condition = train_data[(i * batch_size):(i * batch_size + curr_batch_size), fea:fea + l]
                real = train_data[(i * batch_size):(i * batch_size + curr_batch_size), fea + l:fea + l + pred]

                feature_data = feature_data.unsqueeze(0).to(device).to(torch.float)
                condition = condition.unsqueeze(0).to(device).to(torch.float)
                real = real.unsqueeze(0).to(device).to(torch.float)

                ### Update Discriminator ###
                for j in range(diter):
                    disc_opt.zero_grad()

                    # ✅ เพิ่ม Feature เข้าไปใน Input ของ Generator
                    noise = torch.randn(1, curr_batch_size, z_dim, device=device, dtype=torch.float)
                    fake = gen(noise, feature_data, condition, h_0g, c_0g)

                    # ✅ รวม Feature + Condition + Fake/Real
                    fake_and_condition = combine_vectors(feature_data, condition, fake, dim=-1)
                    real_and_condition = combine_vectors(feature_data, condition, real, dim=-1)

                    disc_fake_pred = disc(fake_and_condition.detach(), h_0d, c_0d)
                    disc_real_pred = disc(real_and_condition, h_0d, c_0d)

                    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
                    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
                    disc_loss = (disc_fake_loss + disc_real_loss) / 2

                    disc_loss.backward()
                    disc_opt.step()

                ### Update Generator ###
                gen_opt.zero_grad()
                noise = torch.randn(1, curr_batch_size, z_dim, device=device, dtype=torch.float)
                fake = gen(noise, feature_data, condition, h_0g, c_0g)
                fake_and_condition = combine_vectors(feature_data, condition, fake, dim=-1)

                disc_fake_pred = disc(fake_and_condition, h_0d, c_0d)

                ft = fake.squeeze(0).squeeze(1)
                rl = real.squeeze(0).squeeze(1)

                sign_approx = torch.tanh(tanh_coeff * ft)
                PnL_s = sign_approx * rl
                PnL = torch.mean(PnL_s)
                MSE = (torch.norm(ft - rl) ** 2) / curr_batch_size
                SR = (torch.mean(PnL_s)) / (torch.std(PnL_s))
                STD = torch.std(PnL_s)

                gen_opt.zero_grad()
                SR.backward(retain_graph=True)
                SR_norm[epoch * nbatches + i] = get_gradient_norm(gen)

                gen_opt.zero_grad()
                PnL.backward(retain_graph=True)
                PnL_norm[epoch * nbatches + i] = get_gradient_norm(gen)

                gen_opt.zero_grad()
                MSE.backward(retain_graph=True)
                MSE_norm[epoch * nbatches + i] = get_gradient_norm(gen)

                gen_opt.zero_grad()
                STD.backward(retain_graph=True)
                STD_norm[epoch * nbatches + i] = get_gradient_norm(gen)

                gen_opt.zero_grad()
                gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
                gen_loss.backward()
                BCE_norm[epoch * nbatches + i] = get_gradient_norm(gen)
                gen_opt.step()

        alpha = torch.mean(BCE_norm / PnL_norm)
        beta = torch.mean(BCE_norm / MSE_norm)
        gamma = torch.mean(BCE_norm / SR_norm)
        delta = torch.mean(BCE_norm / STD_norm)

        print("Completed.")
        print(r"$\alpha$:", alpha)
        print(r"$\beta$:", beta)
        print(r"$\gamma$:", gamma)
        print(r"$\delta$:", delta)

        return gen, disc, gen_opt, disc_opt, alpha, beta, gamma, delta

    start_time_gc = time.time()
    print("Start GradientCheck -------------------------------------------------")
    gen_gc1, disc_gc1, gen_opt, disc_opt, alpha, beta, gamma, delta = GradientCheck( gen, disc, gen_opt, disc_opt, criterion, n_epochs, train_data,batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, fea, pred, diter, tanh_coeff, device, plot)
    print("End GradientCheck -------------------------------------------------")
    end_time_gc = time.time()

    def TrainLoopMainSRMSEnv_val(
        gen, disc, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch,
        train_data, validation_data, batch_size, hid_d, hid_g, z_dim, lr_d=0.0001, lr_g=0.0001, h=1, l=10, fea=19, pred=1,
        diter=1, tanh_coeff=100, device=device, plot=False
    ):
        """
        Training loop with validation: PnL, MSE, SR loss (with additional features)
        """
        ntrain = train_data.shape[0]
        nval = validation_data.shape[0]
        nbatches = ntrain // batch_size + 1
        best_val_loss = float('inf')
        best_gen_state = None
        best_disc_state = None
        dropout_rate=0.2  # Added dropout rate parameter
        early_stopping_patience = 30  # Added early stopping patience
        l2_lambda=1e-4  # Added L2 regularization parameter
        # Early stopping setup
        patience_counter = 0

        # early_stopping_patience = int(early_stopping_patience)
        # Learning rate schedulers
        gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(gen_opt, mode='min', factor=0.5, patience=3, verbose=True)
        disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(disc_opt, mode='min', factor=0.5, patience=3, verbose=True)

        # Training history
        train_losses = []
        val_losses = []

        gen.train()
        for epoch in tqdm(range(n_epochs)):
            epoch_train_losses = []
            perm = torch.randperm(ntrain)
            train_data = train_data[perm, :]

            for i in range(nbatches):
                curr_batch_size = batch_size if i != (nbatches - 1) else ntrain - i * batch_size
                h_0d = torch.zeros((1, curr_batch_size, hid_d), device=device, dtype=torch.float)
                c_0d = torch.zeros((1, curr_batch_size, hid_d), device=device, dtype=torch.float)
                h_0g = torch.zeros((1, curr_batch_size, hid_g), device=device, dtype=torch.float)
                c_0g = torch.zeros((1, curr_batch_size, hid_g), device=device, dtype=torch.float)

                # ✅ **ดึงค่า Feature และ Condition (Feature + l ค่า Return ก่อนหน้า)**
                feature_data = train_data[(i * batch_size):(i * batch_size + curr_batch_size), 0:fea]
                condition = train_data[(i * batch_size):(i * batch_size + curr_batch_size), fea:fea + l]
                real = train_data[(i * batch_size):(i * batch_size + curr_batch_size), fea + l:fea + l + pred]

                feature_data = feature_data.unsqueeze(0).to(device).to(torch.float)
                condition = condition.unsqueeze(0).to(device).to(torch.float)
                real = real.unsqueeze(0).to(device).to(torch.float)

                ### Update discriminator ###
                for _ in range(diter):
                    disc_opt.zero_grad()
                    noise = torch.randn(1, curr_batch_size, z_dim, device=device, dtype=torch.float)
                    fake = gen(noise, feature_data, condition, h_0g, c_0g)

                    # ✅ **รวม Feature + Condition + Fake/Real**
                    fake_and_condition = combine_vectors(feature_data, condition, fake, dim=-1)
                    real_and_condition = combine_vectors(feature_data, condition, real, dim=-1)

                    disc_fake_pred = disc(fake_and_condition.detach(), h_0d, c_0d)
                    disc_real_pred = disc(real_and_condition, h_0d, c_0d)
                        # Add noise to discriminator input (label smoothing)
                    real_labels = torch.ones_like(disc_fake_pred).to(device) * 0.9  # Smooth positive labels to 0.9
                    fake_labels = torch.zeros_like(disc_fake_pred).to(device) + 0.1  # Smooth negative labels to 0.1

                        # Add L2 regularization for discriminator
                    l2_reg_d = 0
                    for param in disc.parameters():
                        l2_reg_d += torch.norm(param)

                    disc_fake_loss = criterion(disc_fake_pred, fake_labels)
                    disc_real_loss = criterion(disc_real_pred, real_labels)
                    disc_loss = (disc_fake_loss + disc_real_loss) / 2 + l2_lambda * l2_reg_d
                    # disc_loss = (disc_fake_loss + disc_real_loss) / 2

                    disc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)  # Gradient clipping
                    disc_opt.step()

                ### Update generator ###
                gen_opt.zero_grad()
                noise = torch.randn(1, curr_batch_size, z_dim, device=device, dtype=torch.float)
                fake = gen(noise, feature_data, condition, h_0g, c_0g)
                fake_and_condition = combine_vectors(feature_data, condition, fake, dim=-1)

                disc_fake_pred = disc(fake_and_condition, h_0d, c_0d)

                # Add L2 regularization for generator
                l2_reg_g = 0
                for param in gen.parameters():
                    l2_reg_g += torch.norm(param)

                sign_approx = torch.tanh(tanh_coeff * fake.squeeze())
                PnL_s = sign_approx * real.squeeze()
                PnL, SqLoss, SR = torch.mean(PnL_s), (torch.norm(fake.squeeze() - real.squeeze()) ** 2) / curr_batch_size, torch.mean(PnL_s) / torch.std(PnL_s)
                gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred)) + beta * SqLoss - gamma * SR

                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
                gen_opt.step()

                epoch_train_losses.append(gen_loss.item())

            # Calculate average training loss for the epoch
            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.append(avg_train_loss)

            ### Validation ###
            gen.eval()
            with torch.no_grad():
                val_losses = []
                epoch_val_losses = []
                for j in range(0, nval, batch_size):
                    curr_batch_size = min(batch_size, nval - j)
                    h_0d = torch.zeros((1, curr_batch_size, hid_d), device=device, dtype=torch.float)
                    c_0d = torch.zeros((1, curr_batch_size, hid_d), device=device, dtype=torch.float)
                    h_0g = torch.zeros((1, curr_batch_size, hid_g), device=device, dtype=torch.float)
                    c_0g = torch.zeros((1, curr_batch_size, hid_g), device=device, dtype=torch.float)

                    # feature_data = validation_data[j:j + curr_batch_size, :fea]  # 19 Features
                    # condition = validation_data[j:j + curr_batch_size, fea:fea + l]  # l ค่า Return ก่อนหน้า
                    # condition = torch.cat((feature_data, condition), dim=-1).unsqueeze(0)
                    # real = validation_data[j:j + curr_batch_size, fea + l:fea + l + pred].unsqueeze(0)

                    feature_data = validation_data[j:j + curr_batch_size, :fea]
                    condition = validation_data[j:j + curr_batch_size, fea:fea + l]
                    real = validation_data[j:j + curr_batch_size, fea + l:fea + l + pred]

                    feature_data = feature_data.unsqueeze(0).to(device).to(torch.float)
                    condition = condition.unsqueeze(0).to(device).to(torch.float)
                    real = real.unsqueeze(0).to(device).to(torch.float)

                    # noise = torch.randn(1, curr_batch_size, z_dim, device=device)
                    # fake = gen(noise, condition, h_0g, c_0g)
                    # น้อยๆ
                    noise = torch.randn(1, curr_batch_size, z_dim, device=device, dtype=torch.float)
                    fake = gen(noise, feature_data, condition, h_0g, c_0g)

                    sign_approx = torch.tanh(tanh_coeff * fake.squeeze())
                    PnL_s = sign_approx * real.squeeze()
                    val_loss = beta * (torch.norm(fake.squeeze() - real.squeeze()) ** 2) / curr_batch_size - gamma * (torch.mean(PnL_s) / torch.std(PnL_s))
                    epoch_val_losses.append(val_loss.item())

                avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
                val_losses.append(avg_val_loss)

                # avg_val_loss = sum(val_losses) / len(val_losses)

                # Learning rate scheduling
                gen_scheduler.step(avg_val_loss)
                disc_scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_gen_state = gen.state_dict()
                    best_disc_state = disc.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

                    # Print epoch statistics
            if (epoch + 1) % 5 == 0:  # Print every 5 epochs
                print(f"Epoch [{epoch+1}/{n_epochs}]")
                print(f"Training Loss: {avg_train_loss:.4f}")
                print(f"Validation Loss: {avg_val_loss:.4f}")

            gen.train()

        if best_gen_state is not None:
            gen.load_state_dict(best_gen_state)
            disc.load_state_dict(best_disc_state)

        print("Training completed. Best validation loss:", best_val_loss)
        return gen, disc, gen_opt, disc_opt

    start_time_main = time.time()
    print("Start TrainLoopMain -------------------------------------------------")
    genPnLMSESR, discPnLMSESR, gen_optPnLMSESR, disc_optPnLMSESR = TrainLoopMainSRMSEnv_val(gen_gc1, disc_gc1, gen_opt, disc_opt, criterion, alpha, beta, gamma, delta, n_epochs, checkpoint_epoch, train_data, validation_data, batch_size,hid_d, hid_g, z_dim, lrd, lrg, h, l, fea, pred, diter, tanh_coeff, device,  plot)
    print("End TrainLoopMain -------------------------------------------------")
    end_time_main = time.time()

    # print(f"TrainLoopMainSRMSEnv_val Time: {time.time() - start_time} seconds")

    def calculate_sharpe_ratio(mean_pnl,std_pnl, risk_free_rate=0.021, trading_days=252):
        """
        คำนวณ Annualized Sharpe Ratio จาก PnL รายวัน
        :param pnl_series: torch.Tensor ของ PnL รายวัน
        :param risk_free_rate: อัตราผลตอบแทนไร้ความเสี่ยง
        :อัตราผลตอบแทนพันธบัตร 1 ปี ณ วันที่ 26/10/2023 คือ 2.21%
        :param trading_days: จำนวนวันซื้อขายใน 1 ปี (ค่าเริ่มต้น 252 วัน)
        :return: ค่า Sharpe Ratio ที่ปรับเป็นรายปีแล้ว
        """

        # คำนวณ Sharpe Ratio รายวัน
        daily_risk_free = risk_free_rate / trading_days  # คิดเป็นรายวัน
        sharpe_daily = (mean_pnl - daily_risk_free) / std_pnl

        # Annualized Sharpe Ratio
        sharpe_annualized = sharpe_daily * np.sqrt(trading_days)

        return sharpe_annualized.item(),sharpe_daily


    def Evaluation2(ticker, gen, test_data, val_data, h, l, pred, fea, hid_d, hid_g, z_dim, lrg, lrd, n_epochs, losstype, sr_val, device, plotsloc, f_name, plot=False):

        df_temp = False
        dt = {'lrd':lrd,'lrg':lrg,'type': losstype,'epochs':n_epochs, 'ticker':ticker,  'hid_g':hid_g, 'hid_d':hid_d}

        ntest = test_data.shape[0]
        gen.eval()
        with torch.no_grad():
            print("test_data")
            print("Shape:", test_data.shape)  # ขนาดของข้อมูล (rows, columns)
            print("Data Type:", test_data.dtype)  # ประเภทของข้อมูล
            print("First 5 Rows:\n", test_data[:5])  # ดูตัวอย่างข้อมูล

            feature_data = test_data[:, :fea]  # ดึง features
            condition_returns = test_data[:, fea:fea+l]  # ดึง historical returns

            # ปรับ dimension และ device
            feature_data = feature_data.unsqueeze(0).to(device).to(torch.float)
            condition_returns = condition_returns.unsqueeze(0).to(device).to(torch.float)

            ntest = test_data.shape[0]
            h0 = torch.zeros((1, ntest, hid_g), device=device, dtype=torch.float)
            c0 = torch.zeros((1, ntest, hid_g), device=device, dtype=torch.float)

            # สร้าง predictions
            fake_noise = torch.randn(1, ntest, z_dim, device=device, dtype=torch.float)
            fake1 = gen(fake_noise, feature_data, condition_returns, h0, c0)

            fake1 = fake1.unsqueeze(0).unsqueeze(2)
            generated1 = torch.empty([1,1,1,ntest,1000])
            generated1[0,0,0,:,0] = fake1[0,0,0,:,0].detach()

            for i in range(999):
                fake_noise = torch.randn(1, ntest, z_dim, device=device, dtype=torch.float)
                fake1 = gen(fake_noise, feature_data, condition_returns, h0, c0)
                fake1 = fake1.unsqueeze(0).unsqueeze(2)
                generated1[0, 0, 0, :, i+1] = fake1[0,0,0,:,0].detach()
                del fake1
                del fake_noise

        b1 = generated1.squeeze()
        mn1 = torch.mean(b1, dim=1)
        real1 = test_data[:, -1]
        print("Real values shape:", real1.shape)

        rl1 = real1.squeeze()
        rmse1 = torch.sqrt(torch.mean((mn1.to(device) - rl1.to(device))**2))
        mae1 = torch.mean(torch.abs(mn1.to(device)-rl1.to(device)))
        dt['RMSE'] = rmse1.item()
        dt['MAE'] = mae1.item()
        ft1 = mn1.clone().detach().to(device)
        PnL1,meanPnL,std_pnl,cumulative_PnL,sgn_fake,forecast = getPnL(stock,ft1,rl1,ntest)
        # print(f"cumulative_PnL - {cumulative_PnL}")
        cumulative_PnL_list = cumulative_PnL.tolist()

        with open(f"Factor-Fin-GAN_cut-{stock}_cumulative_pnl.json", "w") as f:
            json.dump(cumulative_PnL_list, f)


        print("RMSE: ",rmse1,"MAE: ",mae1)
        print("PnL in bp", PnL1)
        # คำนวณ Sharpe Ratio
        sharpe_ratio,sharpe_ratio_dayli = calculate_sharpe_ratio(meanPnL,std_pnl)
        print(f"Dayli Sharpe Ratio: {sharpe_ratio_dayli:.2f}")
        print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")

        #look at the Sharpe Ratio
        n_b1 = b1.shape[1]
        PnL_ws1 = torch.empty(ntest)
        for i1 in range(ntest):
            fk1 = b1[i1,:]
            pu1 = (fk1>=0).sum()
            pu1 = pu1/n_b1
            pd1 = 1-pu1
            PnL_temp1 = 10000*(pu1*rl1[i1].item()-pd1*rl1[i1].item())
            PnL_ws1[i1] = PnL_temp1.item()
        PnL_ws1 = np.array(PnL_ws1)
        PnL_wd1 = np.zeros(int(0.5 * len(PnL_ws1)))
        PnL_even = np.zeros(int(0.5 * len(PnL_ws1)))
        PnL_odd = np.zeros(int(0.5 * len(PnL_ws1)))
        for i1 in range(len(PnL_wd1)):
            PnL_wd1[i1] = PnL_ws1[2 * i1] + PnL_ws1[2 * i1 + 1]
            PnL_even[i1] = PnL_ws1[2 * i1]
            PnL_odd[i1] = PnL_ws1[2 * i1 + 1]
        PnL_test = PnL_wd1
        PnL_w_m1 = np.mean(PnL_wd1)
        PnL_w_std1 = np.std(PnL_wd1)
        SR1 = PnL_w_m1/PnL_w_std1
        #print("Sharpe Ratio: ",SR)
        dt['SR_w scaled'] = SR1*np.sqrt(252)
        dt['PnL_w'] = PnL_w_m1

        distcheck = np.array(b1[1,:].cpu())
        means = np.array(mn1.detach())
        reals = np.array(rl1.detach().cpu())
        sgn_fake = np.array(sgn_fake.detach().cpu())
        forecast = np.array(forecast.detach().cpu())
        dt['Corr'] = np.corrcoef([means,reals])[0,1]
        dt['Pos mn'] = np.sum(means >0)/ len(means)
        dt['Neg mn'] = np.sum(means <0)/ len(means)
        print('Correlation ',np.corrcoef([means,reals])[0,1] )

        dt['narrow dist'] = (np.std(distcheck)<0.0002)

        means_gen = means
        reals_test = reals
        distcheck_test = distcheck
        rl_test = reals[1]

        mn = torch.mean(b1,dim=1)
        mn = np.array(mn.cpu())
        dt['narrow means dist'] = (np.std(mn)<0.0002)

        df_temp = pd.DataFrame(data=dt,index=[0])

        return df_temp, rmse1, mae1, PnL1, sharpe_ratio, sharpe_ratio_dayli,sgn_fake,forecast

    df_temp, rmse1, mae1, PnL1, sharpe_ratio, sharpe_ratio_dayli,sgn_fake,forecast = Evaluation2(
        ticker=ticker,
        gen=genPnLMSESR,
        test_data=test_data,
        val_data=validation_data,
        h=h,
        l=l,
        pred=pred,
        fea=fea,  # จำนวน features
        hid_d=hid_d,
        hid_g=hid_g,
        z_dim=z_dim,
        lrg=lrg,
        lrd=lrd,
        n_epochs=n_epochs,
        losstype="PnL MSE SR",
        sr_val=0,
        device=device,
        plotsloc="",  # หรือระบุ path ที่ต้องการ
        f_name="",  # หรือระบุชื่อไฟล์ที่ต้องการ
        plot=False
    )
    from memory_profiler import memory_usage
    from functools import partial
    rmse = rmse1.item()
    mae = mae1.item()
    PnL = PnL1.item()
    sharpe_ratio_dayli = sharpe_ratio_dayli.item()
    sr_facfincut = sharpe_ratio_dayli
    rmse_facfincut = rmse
    sgn_fake1 = sgn_fake
    forecast1 = forecast

    import psutil
    import os
    
    def measure_memory_usage(func):
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func()
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        return result, memory_used
    
    # วัด Memory Usage สำหรับ GradientCheck
    gradient_check_func = lambda: GradientCheck(
        gen, disc, gen_opt, disc_opt, criterion, n_epochs, 
        train_data, batch_size, hid_d, hid_g, z_dim, lrd, lrg, 
        h, l, fea, pred, diter, tanh_coeff, device, plot
    )
    _, mem_usage_gc = measure_memory_usage(gradient_check_func)
    print(f"GradientCheck Memory Usage: {mem_usage_gc:.2f} MB")
    
    # วัด Memory Usage สำหรับ TrainLoopMainSRMSEnv_val
    train_loop_func = lambda: TrainLoopMainSRMSEnv_val(
        gen_gc1, disc_gc1, gen_opt, disc_opt, criterion, 
        alpha, beta, gamma, delta, n_epochs, checkpoint_epoch,
        train_data, validation_data, batch_size, hid_d, hid_g,
        z_dim, lrd, lrg, h, l, fea, pred, diter, tanh_coeff, device, plot
    )
    _, mem_usage_train = measure_memory_usage(train_loop_func)
    print(f"TrainLoopMainSRMSEnv_val Memory Usage: {mem_usage_train:.2f} MB")


    print(f"GradientCheck Processing Time: {end_time_gc - start_time_gc:.4f} seconds")
    print(f"TrainLoopMainSRMSEnv_val Processing Time: {end_time_main - start_time_main:.4f} seconds")

    all_time = (end_time_gc - start_time_gc) + (end_time_main - start_time_main)
    print(f"Processing Time = {all_time:.4f} seconds")
    all_us = mem_usage_gc + mem_usage_train
    print(f"Memory Usage = {all_us:.4f} MB")
    df_result[stock] = [rmse, mae, PnL, sharpe_ratio, sharpe_ratio_dayli,all_time,all_us]

    joblib.dump(genPnLMSESR, f"Factor-Fingan_{stock}_model.pkl")  # เซฟเป็นไฟล์ .pkl

    df = pd.DataFrame(sgn_fake, columns=['sign'])

# บันทึกเป็นไฟล์ CSV
    df.to_csv(f'sign_data_fincut_{stock}.csv', index=False)
    sgn_fake_facfin_cut = sgn_fake
    forecast_facfincut = forecast
    df = pd.DataFrame(forecast, columns=['sign'])

# บันทึกเป็นไฟล์ CSV
    df.to_csv(f'forecast_facfincut_{stock}.csv', index=False)


df_result.to_csv(f"Factor_cut-Fin-GAN_All_result_{st}.csv", index=True)
