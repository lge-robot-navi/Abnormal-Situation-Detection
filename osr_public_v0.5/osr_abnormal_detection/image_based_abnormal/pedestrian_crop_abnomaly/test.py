"""
--------------------------------------------------------------------------
    pedestrian crop image anomaly
    H.C. Shin, creatrix@etri.re.kr, 2019.10.24
--------------------------------------------------------------------------
    Copyright (C) <2019>  <H.C. Shin, creatrix@etri.re.kr>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
--------------------------------------------------------------------------
"""
# --phase 'test' --dataroot "./data" --batchsize 5 --epochs 1 --load_weights --outf "output"
from __future__ import print_function
from data import load_data
from model_4_human import AAE_basic
from options import Options

def main():
    # ARGUMENTS
    opt = Options().parse()

    # LOAD DATA
    dataloader = load_data(opt)

    # LOAD MODEL
    model = AAE_basic(opt, dataloader)

    # TRAIN MODEL
    model.test()

if __name__ == '__main__':
    main()
