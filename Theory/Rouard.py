import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))))
import openpyxl as px
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np

def load_3col_csv(file_dir_name, column_split = '\t', column_name_exist = False ):
    lines = list(filter(None,open(file_dir_name, "r").read().split("\n")))
    if column_name_exist:
        lines = lines[1:]
    n_col = len(lines[0])
    col = []
    for i in range(n_col):
        col.append(["" for x in range(len(lines))])
    i=0
    for line in lines:
        values = line.split(column_split)
        for j in range(len(values)):
            col[j][i] = values[j]
        i=i+1
    df = {}
    for k in range(n_col):
        df['col'+str(k+1)] = col[k]
    df= pd.DataFrame(df)
    return df

def nk(raw_nk_file, nk_loc = ".\\NK\\"):
    nk_df = load_3col_csv(nk_loc + raw_nk_file, 't', False)
    wavelength = list(nk_df['col1'])
    wavelength = [float(x)*0.1 for x in wavelength]
    n = list(nk_df['col2'])
    n = [float(x) for x in n]
    k = list(nk_df['col3'])
    k = [float(x) for x in k]
    return n,k,wavelength

def find_idx(wavelength, min_value,max_value):
    wl_max_idx = 0
    for wl in wavelength:
        if wl >= min_value:
            wl_min_idx = np.where(wavelength==wl)[0][-1]
            break
    for wl in wavelength[::-1]:
        if wl <= max_value:
            wl_max_idx = np.where(wavelength==wl)[0][-1]
            break
    if wl_min_idx>wl_max_idx:
        print('Error: wl_min_idx > wl_max_idx, index follows min max index of given wavelength')
        return np.where(wavelength==wavelength[0])[0][-1], np.where(wavelength==wavelength[-1])[0][-1]
    else:
        return wl_min_idx,wl_max_idx

def load_1col_xlsx(file_dir_name, sheet = "Sheet1"):
    p = px.load_workbook(file_dir_name)[sheet]
    lamda = []
    for row in p.iter_rows():
        for k in row:
            lamda.append(np.float(k.internal_value))
    return np.array(lamda)

def pre_calculation(layer_list, wave_file, nk_loc, w_min=450, w_max=650):
    n_layer = len(layer_list)
    w = load_1col_xlsx(wave_file)
    min_from = max_from = n_layer
    raw_n = list(range(n_layer))
    raw_k = list(range(n_layer))
    raw_w = list(range(n_layer))
    w_min_idx, w_max_idx = find_idx(w,w_min,w_max)

    for j in range(n_layer):
        raw_n[j], raw_k[j], raw_w[j] = nk(layer_list[j],nk_loc = nk_loc)
        wl_min, wl_max = find_idx(w,raw_w[j][0],raw_w[j][-1])
        if w_min < w[wl_min]:
            w_min = w[wl_min]
            w_min_idx = wl_min
            min_from = j
        if w_max > w[wl_max]:
            w_max = w[wl_max]
            w_max_idx = wl_max
            max_from = j

    if min_from == n_layer:
        str_min_from = 'User setting'
    else:
        str_min_from = layer_list[min_from]
    if max_from == n_layer:
        str_max_from = 'User setting'
    else:
        str_max_from = layer_list[max_from]
    if min_from != n_layer or max_from != n_layer:
        print('\n')
        print('# user can only use w_min and w_max from the range of nk wavelength')
        print('# min max wavelength of nk:',w_min, w_max)
        print('# min from:', str_min_from, ', max from:', str_max_from)
        print('\n')
    w = w[w_min_idx:w_max_idx]

    N = np.zeros((n_layer,w.shape[0]), dtype = np.complex_)
    for j in range(n_layer):
        n = interp1d(raw_w[j], raw_n[j], kind='cubic')(w)
        k = interp1d(raw_w[j], raw_k[j], kind='cubic')(w)
        N[j] = n - 1j*k
    return n_layer, w, N

def s_r(N1,N2,c1,c2):
    return (N1*c1-N2*c2)/(N1*c1+N2*c2)

def s_t(N1,N2,c1,c2):
    return 2*N1*c1/(N1*c1+N2*c2)

def p_r(N1,N2,c1,c2):
    return (N2*c1-N1*c2)/(N2*c1+N1*c2)

def p_t(N1,N2,c1,c2):
    return 2*N1*c1/(N2*c1+N1*c2)

def Rouard(thickness_list, n_layer, w, N, inc_angle, pol_state = 's'):
    costh = np.zeros((n_layer,w.shape[0]), dtype = np.complex_)
    costh[0] = np.array([(inc_angle*np.pi/180) for i in range(w.shape[0])], dtype = np.complex_)
    for j in range(n_layer-1):
        costh[j+1] = np.arcsin(np.divide(N[j],N[j+1])*np.sin(costh[j]))
    costh = np.cos(costh)
    d = np.insert(np.append(np.array(thickness_list),np.zeros((1))),0,0)[::-1]
    N = N[::-1]
    costh = costh[::-1]
    if pol_state == 's':
        f_r = s_r
    elif pol_state == 'p':
        f_r = p_r
    rho = f_r(N[0],N[1],costh[0],costh[1])
    for j in range(1,n_layer-1,1): # 1:j-1, 2:j, ....,
        r = f_r(N[j],N[j+1],costh[j],costh[j+1])
        b = np.exp(-2j*(2 * np.pi* d[j]) * ((N[j] * costh[j])/w))
        rho = (r+rho*b)/(1+r*rho*b)
    rho = np.square(np.absolute(rho))
    return rho

if __name__ == "__main__":
    nk_loc = "..\\Layer_Material_NK\\"
    ref_w = "..\\Reference_files\\Spectrometer_wavelength.xlsx"
    reference = "..\\Reference_files\\Si_substrate_reference_Reflectance.xlsx"
    layer_sample = ["air.nk", "1층막_layer1_SiO2.nk", "Substrate_Si.nk"]
    inc_angle = 0
    pol_state = 's'
    w_min = 450
    w_max = 700
