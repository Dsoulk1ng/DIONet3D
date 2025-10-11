import os, re, gzip
import numpy as np 
from util import instance_direction_rect, save, is_gzip_file
from multiprocessing import Pool, cpu_count
from scipy.ndimage import gaussian_filter
import argparse
import math
from scipy.fft import fft2, ifft2, fftfreq


class Paraser(object):
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        #self.parser.add_argument('--data_root',     default='./original/test/partitionO/', help='data to be processed from EDA tools')
        #self.parser.add_argument('--data_root',     default='./original/test/partitionP/', help='data to be processed from EDA tools')
        #self.parser.add_argument('--data_root',     default='./original/test/RISCY/', help='data to be processed from EDA tools')
        self.parser.add_argument('--data_root',     default='./original/test/zero_riscy/', help='data to be processed from EDA tools')
        
        self.parser.add_argument('--lef_path',      default='./original/tech1_lef/', help='path to tech1 LEF files')

        #self.parser.add_argument('--def_path',      default='./original/test/partitionO/def/', help='path to DEF files')
        #self.parser.add_argument('--def_path',      default='./original/test/partitionP/def/', help='path to DEF files')
        #self.parser.add_argument('--def_path',      default='./original/test/RISCY/def/', help='path to DEF files')
        self.parser.add_argument('--def_path',      default='./original/test/zero_riscy/def/', help='path to DEF files')

        #self.parser.add_argument('--rpt_path',      default='./original/test/partitionO/pwr_rpt/', help='path to power rpt files')
        #self.parser.add_argument('--rpt_path',      default='./original/test/partitionP/pwr_rpt/', help='path to power rpt files')
        #self.parser.add_argument('--rpt_path',      default='./original/test/RISCY/pwr_rpt/', help='path to power rpt files')
        self.parser.add_argument('--rpt_path',      default='./original/test/zero_riscy/pwr_rpt/', help='path to power rpt files')

        #self.parser.add_argument('--bumpfile_path', default='./original/test/partitionO/bumpfile/', help='path to bump locations files')
        #self.parser.add_argument('--bumpfile_path', default='./original/test/partitionP/bumpfile/', help='path to bump locations files')
        #self.parser.add_argument('--bumpfile_path', default='./original/test/RISCY/bumpfile/', help='path to bump locations files')
        self.parser.add_argument('--bumpfile_path', default='./original/test/zero_riscy/bumpfile/', help='path to bump locations files')


        #self.parser.add_argument('--iv_path',       default='./original/test/partitionO/iv/', help='path to instance voltage files')
        #self.parser.add_argument('--iv_path',       default='./original/test/partitionP/iv/', help='path to instance voltage files')
        #self.parser.add_argument('--iv_path',       default='./original/test/RISCY/iv/', help='path to instance voltage files')
        self.parser.add_argument('--iv_path',       default='./original/test/zero_riscy/iv/', help='path to instance voltage files')


        self.parser.add_argument('--save_path',     default='./processed/testing_set', help='processed data input into the network')

        self.parser.add_argument('--unit', default=2000, help='set 2000 when design is test')


def read_lef(path, lef_dict):
    with open(path, 'r') as read_file:
        cell_name = ''
        pin_name = ''
        rect_list_left = []
        rect_list_lower = []
        rect_list_right = []
        rect_list_upper = []
        READ_MACRO = False
        for line in read_file:
            if line.lstrip().startswith('MACRO'):
                READ_MACRO = True
                cell_name = line.split()[1]
                lef_dict[cell_name] = {}
                lef_dict[cell_name]['pin'] = {}

            if READ_MACRO:
                if line.lstrip().startswith('SIZE'):
                    l = re.findall(r'-?\d+\.?\d*e?-?\d*?', line)
                    lef_dict[cell_name]['size'] = [float(l[0]), float(l[1])] 

                elif line.lstrip().startswith('PIN'):
                    pin_name = line.split()[1]

                elif line.lstrip().startswith('RECT'):
                    l = line.split()
                    rect_list_left.append(float(l[1]))
                    rect_list_lower.append(float(l[2]))
                    rect_list_right.append(float(l[3]))
                    rect_list_upper.append(float(l[4]))

                elif line.lstrip().startswith('POLYGON'):
                    l = line.split()
                    rect_list_left.append(float(l[1]))
                    rect_list_left.append(float(l[11]))
                    rect_list_left.append(float(l[13]))
                    rect_list_left.append(float(l[15]))
                    rect_list_lower.append(float(l[2]))
                    rect_list_lower.append(float(l[6]))
                    rect_list_lower.append(float(l[14]))
                    rect_list_lower.append(float(l[16]))
                    rect_list_right.append(float(l[3]))
                    rect_list_right.append(float(l[5]))
                    rect_list_right.append(float(l[7]))
                    rect_list_right.append(float(l[9]))
                    rect_list_upper.append(float(l[4]))
                    rect_list_upper.append(float(l[8]))
                    rect_list_upper.append(float(l[10]))
                    rect_list_upper.append(float(l[12]))
                
                elif line.lstrip().startswith('END %s\n' % pin_name):
                    rect_left = min(rect_list_left) 
                    rect_lower = min(rect_list_lower) 
                    rect_right = max(rect_list_right) 
                    rect_upper = max(rect_list_upper) 
                    lef_dict[cell_name]['pin'][pin_name] = [rect_left, rect_lower, rect_right, rect_upper] # pin_rect
                    rect_list_left = []
                    rect_list_lower = []
                    rect_list_right = []
                    rect_list_upper = []

    return lef_dict
def poisson_solver_fft(source, dx=1.0, dy=1.0):
    
    ny, nx = source.shape
    kx = fftfreq(nx, d=dx) * 2 * np.pi
    ky = fftfreq(ny, d=dy) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky, indexing='xy')
    k_squared = kx**2 + ky**2
    k_squared[0, 0] = 1  

    rho_hat = fft2(source)
    V_hat = rho_hat / k_squared
    V_hat[0, 0] = 0  

    potential = np.real(ifft2(V_hat))
    return potential

class ReadInnovusOutput:

    def __init__(self, root_dir, arg, save_name, lef_dict):
        # Only initialize variables if they have not been set before
        if not hasattr(self, 'initialized'):
            self._initialize(root_dir, arg, save_name, lef_dict)
            self.initialized = True

    def _initialize(self, root_dir, arg, save_name, lef_dict):
    
        self.save_name = save_name
        self.save_path = arg.save_path
        self.unit = arg.unit
        
        self.patch_size = [256, 256]

        self.step_x = None
        self.step_y = None

        self.lef_dict = lef_dict 
        
        self.root_dir = root_dir 
        self.route_def_path = os.path.join(root_dir, 'def', save_name + '.def')
        self.power_path = os.path.join(root_dir, 'pwr_rpt', save_name + '.rpt')
        self.bumpfile_path = os.path.join(root_dir, 'bumpfile', save_name + '.loc')
        self.iv_path = os.path.join(root_dir, 'iv', save_name + '.iv')
        
        self.die_area = None                   
        self.route_instance_dict = {}   
        self.route_net_dict = {}        
        self.route_pin_dict = {}        

        self.power_dict = {}
        self.iv_dict = {}
        self.pdn_dict = {}

        self.power_t = None
        self.power_i = None
        self.power_s = None
        self.power_l = None

        self.tsv_list = {}
        self.hb_list = {}
        
    def read_tsv_and_hb_coords(self):

        if is_gzip_file(self.bumpfile_path):
            read_file = gzip.open(self.bumpfile_path,"rt")
        else:
            read_file = open(self.bumpfile_path,"r")
        
        if not self.bumpfile_path.endswith('.loc'):
            print(f"Skippint {self.bumpfile_path}: not a bump file.")
            return

        for line in read_file:
            if line.startswith('TSV:'):
                tokens = line.strip().split()
                if len(tokens) >= 4:
                    net_type = tokens[4]
                    x_coord = float(tokens[2])
                    y_coord = float(tokens[3])
                    if net_type not in self.tsv_list:
                        self.tsv_list[net_type] = []
                    self.tsv_list[net_type].append([x_coord, y_coord])
                    
            elif line.startswith('Bump:'):
                tokens = line.strip().split()
                if len(tokens) >= 7 and tokens[2] == 'UBUMP' and tokens[6] in ('-power', '-ground'):
                    net_type = tokens[5]
                    x_coord = float(tokens[3])
                    y_coord = float(tokens[4])
                    if net_type not in self.hb_list:
                        self.hb_list[net_type] = []
                    self.hb_list[net_type].append([x_coord, y_coord])
                    
        read_file.close()
    

    def read_def(self):
        
        if is_gzip_file(self.route_def_path):
            read_file = gzip.open(self.route_def_path,"rt")
        else:
            read_file = open(self.route_def_path,"r")

        if not self.route_def_path.endswith('.def'):
            print(f"Skippint {self.route_def_path}: not a def file.")
            return
        
        READ_MACROS = False
        READ_NETS = False
        READ_PINS = False
        net = ''
        for line in read_file:
            line = line.lstrip()
            if line.startswith("DIEAREA"):
                die_coordinate = re.findall(r'\d+', line)
                self.die_area = (float(float(die_coordinate[2]) / self.unit), float(float(die_coordinate[3]) / self.unit))
                self.step_x = self.die_area[0] / self.patch_size[0]
                self.step_y = self.die_area[1] / self.patch_size[1]
            if line.startswith("COMPONENTS"):
                READ_MACROS = True
            elif line.startswith("END COMPONENTS"):
                READ_MACROS = False
            elif line.startswith("NETS"):
                READ_NETS =True
            elif line.startswith("END NETS") or line.startswith("SPECIALNETS"):
                READ_NETS = False
            elif line.startswith('PIN'):
                READ_PINS =True
            elif line.startswith('END PINS'):
                READ_PINS = False
            if READ_MACROS :                                            
                if re.search(r'\bFIXED\b', line) or re.search(r'\bPLACED\b', line):
                    words = line.split()
                    if '(' in line:
                        l = words.index('(')
                        instance_name = words[1].replace('\\', '')
                        cell_name = words[2]
                        x_coord = float(float(words[l+1]) / self.unit)
                        y_coord = float(float(words[l+2]) / self.unit)
                        orientation = words[l+4]
                        self.route_instance_dict[instance_name] = [cell_name, (x_coord, y_coord), orientation]

            if READ_NETS:
                if line.startswith('-'):
                    net = line.split()[1].replace('\\', '')             
                    self.route_net_dict[net] = []

                elif line.startswith('('):                              
                    l = line.split()
                    n = 0
                    for k in l:
                        if k == '(':
                            self.route_net_dict[net].append(l[n+1].replace('\\', ''))
                        n += 1
            if READ_PINS:                                               
                if line.startswith('-'):
                    pin = line.split()[1]
                elif line.strip().startswith('+ LAYER'):
                    pin_rect = re.findall(r'\d+', line)
                    self.route_pin_dict[pin] = {}
                    self.route_pin_dict[pin]['layer'] = line.split()[2]
                    self.route_pin_dict[pin]['rect'] = [float(float(pin_rect[-4]) / self.unit), float(float(pin_rect[-3]) / self.unit), float(float(pin_rect[-2]) / self.unit), float(float(pin_rect[-1]) / self.unit)]
                elif line.strip().startswith('+ PLACED') or line.strip().startswith('+ FIXED'):
                    data = line.split()
                    self.route_pin_dict[pin]['location'] = [float(float(data[3]) / self.unit), float(float(data[4]) / self.unit)]
                    self.route_pin_dict[pin]['direction'] = data[6]

        print(f"Finish route def {self.route_def_path} file reading")
        
        read_file.close()
    
    def read_def_pdn(self):

        READ = False
        net_type = None
        if not self.route_def_path.endswith('.def'):
            print(f"Skippint {self.route_def_path}: not a def file.")
            return

        with open(self.route_def_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('- VDD'):
                    READ = True
                    net_type = 'VDD'
                    continue
                elif line.startswith('- VSS'):
                    READ = True
                    net_type = 'VSS'
                    continue

                if READ:
                    if line.startswith(';'):
                        READ = False
                        continue
                    if 'STRIPE' not in line:
                        continue
                    tokens = line.split()
                    if len(tokens) < 6:
                        continue
                    if tokens[2] == '0':
                        continue
                    offset = 1 if tokens[0] == '+' else 0
                    layer = tokens[1 + offset]  # 如 M10、M5
                    width = float(float(tokens[2 + offset]) / self.unit)

                    coords = re.findall(r"\(\s*([-\d\*\.]+)\s+([-\d\*\.]+)\s*\)", line)

                    x1_str, y1_str = coords[0]
                    w_str, h_str = coords[1]

                    x1 = float(float(x1_str) / self.unit)
                    y1 = float(float(y1_str) / self.unit)
                    w = float(w_str) if w_str != '*' else 0
                    h = float(h_str) if h_str != '*' else 0

                    if coords[1][0] == '*':  
                        x1 = float(x1 - width/2)
                        x2 = float(x1 + width/2)
                        y2 = float(h / self.unit)
                    elif coords[1][1] == '*':  
                        y1 = float(y1 - width/2)
                        x2 = float(w / self.unit)
                        y2 = float(y1 + width/2)
                    else:
                        continue
                    
                    if net_type not in self.pdn_dict:
                        self.pdn_dict[net_type] = {}
                    if layer not in self.pdn_dict[net_type]:
                        self.pdn_dict[net_type][layer] = []
                    self.pdn_dict[net_type][layer].append((x1, y1, x2, y2))
    
    def read_power(self):
        if not self.power_path.endswith('.rpt'):
            print(f"Skippint {self.power_path}: not a power rpt file.")
            return
        
        with open(self.power_path, 'r') as read_file:
            start = False
            read = False
            for line in read_file:
                if "Name" in line:
                    start = True
                if start and line.startswith('Total'):
                    break
                if start:
                    if line.startswith('------------'):
                        read = True
                    if read:
                        if not line.strip():
                            continue
                        data = line.split()
                        if len(data) < 6:
                            continue

                        name = data[0].replace('\\', '')
                        int_power = float(data[1]) if data[1] else 0.0
                        sw_power = float(data[2]) if data[2] else 0.0
                        tot_power = float(data[3]) if data[3] else 0.0
                        leak_power = float(data[4]) if data[4] else 0.0
                        self.power_dict[name] = [tot_power, int_power, sw_power, leak_power]
        print(f"Finish {self.power_path} power rpt reading")                
                
    def read_iv(self):
        if not self.iv_path.endswith('.iv'):
            print(f"Skippint {self.iv_path}: not a iv file.")
            return
        
        with open(self.iv_path, 'r') as read_file:
            start = False
            for line in read_file:
                if "BEGIN" in line:
                    start = True
                if start and line.startswith('END'):
                    break
                if start:
                    if not line.strip():
                        continue
                    data = line.split()
                    if len(data) < 6:
                        continue
                    
                    
                    name = data[1].replace('\\', '')
                    if data[2] not in ('NA', '', None):
                        div = float(data[2])
                    else:
                        continue
                    
                    self.iv_dict[name] = div
        print(f"Finish {self.iv_path} iv reading")   

    def apply_tsv_mask(self):
        
        W = self.patch_size[0]
        H = self.patch_size[1]
        
        tsv_mask = np.zeros((H, W))
        top_pdn_mask = np.zeros((H, W))
        pattern = re.compile(r"curdl", re.IGNORECASE) 

        for net_type in ["VDD", "VSS"]:
            for layer, segments in self.pdn_dict.get(net_type, {}).items():
                if not pattern.search(layer):  
                    continue
                for x1, y1, x2, y2 in segments:
                    px1 = min(max(int(x1 / self.step_x), 0), W - 1)
                    py1 = min(max(int(y1 / self.step_y), 0), H - 1)
                    px2 = min(max(int(x2 / self.step_x), 0), W - 1)
                    py2 = min(max(int(y2 / self.step_y), 0), H - 1)

                    x_min_, x_max_ = sorted([px1, px2])
                    y_min_, y_max_ = sorted([py1, py2])
                    top_pdn_mask[y_min_:y_max_+1, x_min_:x_max_+1] = True


        for net_type, coords in self.tsv_list.items():
            for x, y in coords:
                cx = int(x / self.step_x)
                cy = int(y / self.step_y)
                cx = min(max(cx, 0), W - 1)
                cy = min(max(cy, 0), H - 1)
                if top_pdn_mask[cy, cx]:
                    tsv_mask[cy, cx] += 1

        
        rho = gaussian_filter(tsv_mask, sigma=1.5)

        tsv_density = poisson_solver_fft(-rho)

        tsv_density = -tsv_density
        tsv_density -= np.min(tsv_density)
        tsv_density /= np.max(tsv_density) if np.max(tsv_density) > 0 else 1.0

        tsv_density_top = 0.2 * tsv_density  
        tsv_density_bot = tsv_density

        return tsv_density_top, tsv_density_bot

    def apply_hb_mask(self):

        W = self.patch_size[0]
        H = self.patch_size[1]
        hb_mask = np.zeros((H, W))
        top_pdn_mask = np.zeros((H, W))
        pattern = re.compile(r"curdl", re.IGNORECASE) 


        for net_type in ["VDD", "VSS"]:
            for layer, segments in self.pdn_dict.get(net_type, {}).items():
                if not pattern.search(layer):  
                    continue
                for x1, y1, x2, y2 in segments:
                    px1 = min(max(int(x1 / self.step_x), 0), W - 1)
                    py1 = min(max(int(y1 / self.step_y), 0), H - 1)
                    px2 = min(max(int(x2 / self.step_x), 0), W - 1)
                    py2 = min(max(int(y2 / self.step_y), 0), H - 1)

                    x_min_, x_max_ = sorted([px1, px2])
                    y_min_, y_max_ = sorted([py1, py2])
                    top_pdn_mask[y_min_:y_max_+1, x_min_:x_max_+1] = True

        for net_type, coords in self.hb_list.items():
            for x, y in coords:
                cx = int(float(x) / self.step_x)
                cy = int(float(y) / self.step_y)
                cx = min(max(cx, 0), W - 1)
                cy = min(max(cy, 0), H - 1)
                if top_pdn_mask[cy, cx]:
                    hb_mask[cy, cx] += 1
        
        
        rho = gaussian_filter(hb_mask, sigma=1.5)

        hb_density = poisson_solver_fft(-rho)

        hb_density = -hb_density
        hb_density -= np.min(hb_density)
        hb_density /= np.max(hb_density) if np.max(hb_density) > 0 else 1.0

        hb_density_top = hb_density
        hb_density_bot = 0.2 * hb_density  

        return hb_density_top, hb_density_bot  


    def compute_pdn_coverage_map(self, radius=3, sigma=1.0):
        
        W = self.patch_size[0]
        H = self.patch_size[1]
        coverage_map = np.zeros((H, W))

        size = 2 * radius + 1
        kernel = np.zeros((size, size))
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                dist2 = dx**2 + dy**2
                kernel[dy + radius, dx + radius] = math.exp(-dist2 / (2 * sigma**2))

        for net_type in ["VDD", "VSS"]:
            for layer, segments in self.pdn_dict.get(net_type, {}).items():
                for x1, y1, x2, y2 in segments:
                    px1 = min(max(int(x1 / self.step_x), 0), W - 1)
                    py1 = min(max(int(y1 / self.step_y), 0), H - 1)
                    px2 = min(max(int(x2 / self.step_x), 0), W - 1)
                    py2 = min(max(int(y2 / self.step_y), 0), H - 1)

                    x_min_, x_max_ = sorted([px1, px2])
                    y_min_, y_max_ = sorted([py1, py2])



                    for y in range(y_min_, y_max_ + 1):
                        for x in range(x_min_, x_max_ + 1):
                            for dy in range(-radius, radius + 1):
                                for dx in range(-radius, radius + 1):
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < W and 0 <= ny < H:
                                        coverage_map[ny, nx] += kernel[dy + radius, dx + radius]

        coverage_map /= np.max(coverage_map) if np.max(coverage_map) > 0 else 1.0
        return coverage_map[None, :, :]  


    def patch_based_feature_map(self, tsv_map, hb_map):
        
        W = self.patch_size[0]
        H = self.patch_size[1]

        C = 5
        feature_map = np.zeros((C, H, W))
        label_map = np.zeros((H, W))
        counter_map = np.zeros((H, W))

        for name, inst in self.route_instance_dict.items():
            if name not in self.power_dict:
                continue
            x, y = inst[1]
            direction = instance_direction_rect(inst[2])
            size_x, size_y = self.lef_dict[inst[0]]['size']
            x2 = x + size_x * direction[0] + size_y * direction[1]
            y2 = y + size_x * direction[2] + size_y * direction[3]

            is_macro = (abs(x2 - x) > self.step_x) or (abs(y2 - y) > self.step_y)

            if is_macro:
                cx1 = int((min(x, x2)) / self.step_x)
                cx2 = int((max(x, x2)) / self.step_x)
                cy1 = int((min(y, y2) / self.step_y))
                cy2 = int((max(y, y2) / self.step_y))
                cx1, cx2 = max(0, cx1), min(W-1, cx2)
                cy1, cy2 = max(0, cy1), min(H-1, cy2)
            else:
                cx1 = cx2 = int(x / self.step_x)
                cy1 = cy2 = int(y / self.step_y)
                cx1 = cx2 = min(max(cx1, 0), W-1)
                cy1 = cy2 = min(max(cy1, 0), H-1)
            
            pwr = self.power_dict[name]

            for cx in range(cx1, cx2+1):
                for cy in range(cy1, cy2+1):
                    patch_x1 = cx * self.step_x
                    patch_x2 = patch_x1 + self.step_x
                    patch_y1 = cy * self.step_y
                    patch_y2 = patch_y1 + self.step_y
                    overlap_x = max(0, min(patch_x2, max(x, x2)) - max(patch_x1, min(x, x2)))
                    overlap_y = max(0, min(patch_y2, max(y, y2)) - max(patch_y1, min(y, y2)))
                    overlap_area = overlap_x * overlap_y
                    cell_area = abs((x2 - x) * (y2 - y))

                    if is_macro:
                        weight = overlap_area / cell_area
                        
                    else:
                        weight = 1

                    feature_map[0, cy, cx] += pwr[1] * weight #internal power
                    feature_map[1, cy, cx] += pwr[2] * weight #switching power
                    feature_map[2, cy, cx] += pwr[0] * weight #total power
                    #feature_map[2, cy, cx] += pwr[0] * weight #total power
                    feature_map[3, cy, cx] += (size_x * size_y) * weight #density
                    
                    if name in self.iv_dict:
                        label_map[cy, cx] += self.iv_dict[name] 
                        if self.iv_dict[name] != 0:
                            counter_map[cy, cx] += 1
        
        
        for c in [0, 1, 2]:
            feature_map[c] = np.divide(feature_map[c], counter_map, out=np.zeros_like(feature_map[c]), where=counter_map > 0) #average value
        
        for c in [0, 1, 2]:
            pwr_smooth = gaussian_filter(feature_map[c], sigma=0.5)
            pwr_log = np.log1p(pwr_smooth * 2e4)
            max_val = np.percentile(pwr_log, 99.5)
            min_val = np.percentile(pwr_log, 0.5)
            pwr_feat = (pwr_log - min_val) / (max_val - min_val + 1e-8)
            pwr_feat = np.clip(pwr_feat, 0.0, 1.0)
            feature_map[c] = pwr_feat
        

        label_map = np.divide(label_map, counter_map, out=np.zeros_like(label_map), where=counter_map > 0) #average value
        
        feature_map[3] /= (self.step_x * self.step_y) #density

        pdn_map = self.compute_pdn_coverage_map()

        
        feature_map[4] = pdn_map + hb_map + tsv_map

        
        #min_ch4 = np.min(feature_map[4])
        #max_ch4 = np.max(feature_map[4])
        #feature_map[4] = (feature_map[4] - min_ch4) / (max_ch4 - min_ch4 + 1e-6)
        min_ch4 = np.percentile(feature_map[4], 1)
        max_ch4 = np.percentile(feature_map[4], 99)
        feature_map[4] = (feature_map[4] - min_ch4) / (max_ch4 - min_ch4 + 1e-6)

        #feature_map[5] = hb_map + tsv_map
        #feature_map[6] = pdn_map

        save(self.save_path, 'feature', self.save_name, feature_map)
        save(self.save_path, 'label', self.save_name, label_map)
        
        
def process_designs_paired(design_id, activity, arg, lef_dict):
    
    print(f"Begin processing Design: {design_id}, Activity: {activity}")
    top_name = f"top_die_{design_id}"
    bottom_name = f"bottom_die_{design_id}"

    top_name_iv = f"top_die_{design_id}_{activity}"
    bottom_name_iv = f"bottom_die_{design_id}_{activity}"

    top_obj = ReadInnovusOutput(arg.data_root, arg, top_name, lef_dict)
    bottom_obj = ReadInnovusOutput(arg.data_root, arg, bottom_name, lef_dict)

    top_obj.read_def()
    top_obj.read_def_pdn()
    
    bottom_obj.read_def()
    bottom_obj.read_def_pdn()

    top_obj.save_name = top_name_iv
    bottom_obj.save_name = bottom_name_iv

    top_obj.power_path = os.path.join(top_obj.root_dir, 'pwr_rpt', top_name_iv + '.rpt')
    bottom_obj.power_path = os.path.join(bottom_obj.root_dir, 'pwr_rpt', bottom_name_iv + '.rpt')

    top_obj.iv_path = os.path.join(top_obj.root_dir, 'iv', top_name_iv + '.iv')
    bottom_obj.iv_path = os.path.join(bottom_obj.root_dir, 'iv', bottom_name_iv + '.iv')

    top_obj.read_power()
    top_obj.read_iv()

   
    bottom_obj.read_power()
    bottom_obj.read_iv()

    bottom_obj.read_tsv_and_hb_coords()
    top_obj.read_tsv_and_hb_coords()

    tsv_map_top, tsv_map_bot = bottom_obj.apply_tsv_mask()
    hb_map_top, hb_map_bot = top_obj.apply_hb_mask()
    
    top_obj.patch_based_feature_map(tsv_map_top, hb_map_top)
    
    bottom_obj.patch_based_feature_map(tsv_map_bot, hb_map_bot)
    
    print(f"Finish processing Design: {design_id}, Activity: {activity}")


if __name__ == '__main__':
    argp = Paraser()
    arg = argp.parser.parse_args()
    os.makedirs(arg.save_path, exist_ok=True)

    lef_dic = {}

    if os.path.isdir(arg.lef_path):
        lef_files = [os.path.join(arg.lef_path, f) for f in os.listdir(arg.lef_path) if f.endswith('.lef')]
    else:
        lef_files = [arg.lef_path]
    for i in lef_files:
        lef_dic = read_lef(i, lef_dic)
        print(f"Finish lef file {i} reading")

    def_files = os.listdir(os.path.join(arg.data_root, 'def'))
    iv_files = os.listdir(os.path.join(arg.data_root, 'iv'))
    design_activities = {}
    for f in iv_files:
        if f.endswith('.iv'):
            parts = f.split('_die_')[1].rsplit('_', 1)
            design_id = parts[0]
            activity = parts[1].replace('.iv', '')
            if design_id not in design_activities:
                design_activities[design_id] = []
            design_activities[design_id].append(activity)

    tasks = []
    for design_id, activities in design_activities.items():
        for activity in activities:
            tasks.append((design_id, activity, arg, lef_dic))

    max_process = max(1, cpu_count() // 2)

    with Pool(processes = 10) as pool:
        pool.starmap(process_designs_paired, tasks)
    
    