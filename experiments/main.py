# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Steven Spratley
# Date:    01/11/2022
# Purpose: Trains and evaluates models on Unicode Analogies.


# IMPORTS ----------------------------------------------------------------------------------------------------------- #


import sys, time, os, argparse, torch
import numpy as np
from   torch.utils.data import DataLoader
from   dataset_utility import dataset
from   rpm_solver import RPM_Solver
from   tqdm import tqdm
from   collections import defaultdict
from   mrnet import MRNet


# CONSTANTS --------------------------------------------------------------------------------------------------------- #


BATCH_S = 32
RNDSEED = 1
WORKERS = 8
MULTGPU = True
VALEVRY = 5
TSTEVRY = 100
PERCENT = 100
DEVICE  = 1
SILENT  = True


# ARGS -------------------------------------------------------------------------------------------------------------- #


# Possible models:  ['relbase', 'resnet', 'blind']
# Possible rtpaths: ['./dataset_splits/experiment_1',
#                    './dataset_splits/experiment_2'
#                    './dataset_splits/experiment_3'
#                    './dataset_splits/experiment_4']
# Possible setdirs: ['Ar-N-D-S-H', 'Co-N-D-S-H', 'Di-N-D-S-H', 'Pr-N-D-S-H', 'Un-N-D-S-H']  (experiment_1)
#                   ['GLOB-A-N-D-N-H', 'LOC-A-N-D-N-H', 'OBJ-A-N-D-N-H']                    (experiment_2)
#                   ['A-N-D-N-H', 'A-N-D-S-H', 'A-E-D-S-H', 'A-P-D-S-H']                    (experiment_3)
#                   ['challenge']                                                           (experiment_4)
parser = argparse.ArgumentParser(description='model')
parser.add_argument('--model',   type=str)
parser.add_argument('--rtpath',  type=str)
parser.add_argument('--setdir',  type=str)
parser.add_argument('--epochs',  type=int, default=60)
parser.add_argument('--nummodl', type=int, default=3)
parser.add_argument('--kfolds',  type=int, default=5)
args = parser.parse_args()
if args.model=='scl':
    from SCL.analogy.nn.model import SCL
    BATCH_S = 128


# SCRIPT ------------------------------------------------------------------------------------------------------------ #


#Set cuda and enable cudnn.
if not MULTGPU:
    torch.cuda.set_device(DEVICE)
torch.cuda.cudnn_enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
prog_start = time.time()

#Create datasets and their loaders, per fold.
trn_ds,val_ds,tst_ds,ftstds         = [],[],[],[]
trn_ldrs,val_ldrs,tst_ldrs,ftstldrs = [],[],[],[]
pad = args.model in ['mrnet','scl'] #Models that are expecting a traditional RPM format (3x3) need to have padding.
for i in range(args.kfolds):
    #Sort problems in the fold by rule and class.
    path = f'{args.rtpath}/{args.setdir}/fold_{i}'
    ftst_files = [f for f in os.listdir(f'{path}') if f[:4]=='test']
    prob_types = defaultdict(list)
    for f in ftst_files:
        rule,clss = f.split('_')[1:3]
        if '-' in clss:
            clss = clss.split('-')
            clss = '-'.join(clss[:-2]) if clss[-2].isnumeric() else '-'.join(clss[:-1])
        prob_types[rule].append(f)
        prob_types[clss].append(f)
    trn_ds.append(  dataset(path, mode="train", percent=PERCENT, pad=pad))
    val_ds.append(  dataset(path, mode="val",   percent=PERCENT, pad=pad))
    tst_ds.append(  dataset(path, mode="test",  percent=PERCENT, pad=pad))
    ftstds.append([(dataset(path, mode="ftest", percent=PERCENT, f_names=f, pad=pad), t) for t,f in prob_types.items()])
    trn_ldrs.append(  DataLoader(trn_ds[i], batch_size=BATCH_S, shuffle=True,  num_workers=WORKERS))
    val_ldrs.append(  DataLoader(val_ds[i], batch_size=BATCH_S, shuffle=False, num_workers=WORKERS))
    tst_ldrs.append(  DataLoader(tst_ds[i], batch_size=BATCH_S, shuffle=False, num_workers=WORKERS))
    ftstldrs.append([(DataLoader(d,         batch_size=BATCH_S, shuffle=False, num_workers=WORKERS), t) for d,t in ftstds[i]])

#Define train, validate, and test functions.
def train(model, epoch, ldr):
    model.train()
    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    enum = enumerate(ldr) if SILENT else tqdm(enumerate(ldr))
    for batch_idx, (image, target) in enum:
        counter  += 1
        image     = image.cuda()
        target    = target.cuda()
        loss, acc = model.train_(image, target)
        loss_all += loss
        acc_all  += acc
    if not SILENT:
        print("Epoch {}: Avg Training Loss: {:.6f}".format(epoch, loss_all/float(counter)))
    return loss_all/float(counter)

def validate(model, epoch, ldr):
    model.eval()
    loss_all = 0.0
    acc_all = 0.0
    counter = 0
    for batch_idx, (image, target) in enumerate(ldr):
        counter   += 1
        image     = image.cuda()
        target    = target.cuda() 
        loss, acc = model.validate_(image, target)
        loss_all += loss
        acc_all  += acc
    if not SILENT:
        print("Total Validation Loss: {:.6f}, Acc: {:.4f}".format(loss_all/float(counter), acc_all/float(counter)))
    return loss_all/float(counter), acc_all/float(counter)

def test(model, epoch, ldr):
    model.eval()
    start = time.time()
    acc_all = 0.0
    counter = 0
    for batch_idx, (image, target) in enumerate(ldr):
        counter += 1
        image    = image.cuda()
        target   = target.cuda()
        acc_all += model.test_(image, target)
    average = acc_all/float(counter)
    end = time.time()
    if not SILENT:
        print("Acc: {:.4f}. Tested in {:.2f} seconds.".format(average, end-start))
    return average

def final_test(model, ldrs, top_k=0, print_all=False):
    model.eval()
    ptype_acc = {}
    acc_overall = 0
    for ldr,prob_type in ldrs:
        acc_all = 0.0
        counter = 0
        for batch_idx, (image, target) in enumerate(ldr):
            counter += 1
            image    = image.cuda()
            target   = target.cuda()
            acc_all += model.test_(image, target)
        average = acc_all/float(counter)
        ptype_acc[prob_type] = [average]
        acc_overall += average
    average_overall = acc_overall/len(ldrs)
    return ptype_acc,average_overall

def add_breakdowns(bd1, bd2):
    bd3 = {}
    all_items = list(bd1.items()) + list(bd2.items())
    for ptype,acc_list in all_items:
        bd3[ptype] = acc_list if ptype not in bd3 else bd3[ptype]+acc_list
    return bd3

def avg_breakdowns(bd):
    l = []
    for ptype,accs in bd.items():
        avg_acc = round(sum(accs)/len(accs),2)
        l.append((ptype, avg_acc))
    return l
        

# MAIN -------------------------------------------------------------------------------------------------------------- #


#Train and evaluate num_models per k_fold on the specified dataset split.
def main():
    results_file = open(f'./test_results/{args.setdir}_{args.model}', 'w')
    header = f'Training {args.nummodl}x "{args.model}" models per fold, over {args.kfolds} folds of dataset {args.setdir}.'
    results_file.write(header)
    print(header)
    test_results   = []
    test_breakdown = {}
    
    for i in range(args.kfolds):
        for j in range(args.nummodl):
            #Set seeds and initialise current model.
            np.random.seed(RNDSEED+j)
            torch.manual_seed(RNDSEED+j)
            torch.cuda.manual_seed(RNDSEED+j)
            if args.model=='mrnet':
                model = MRNet().cuda()
            elif args.model=='scl':
                model = SCL().cuda()
            else:
                model = RPM_Solver(args.model, MULTGPU, 3e-4).cuda()
            results_file.write(f'\n    - Fold {i}, model {j}. {len(trn_ds[i].file_names)} train pmps, {len(tst_ds[i].file_names)} test pmps.')
            
            train_start = time.time()
            for epoch in tqdm(range(0, args.epochs), desc=f'   - Fold {i}, model {j}. Epochs'):
                tl = train(model, epoch, trn_ldrs[i])
                # print(f'Trn loss from epoch {epoch}: {round(tl, 2)}')
                if epoch and not epoch % TSTEVRY:
                    breakdown,epoch_acc = final_test(model, ftstldrs[i])
                    print(f'Tst acc from epoch {epoch}: {round(epoch_acc, 1)}')
                    # print(f'Breakdown: {breakdown}')
            model_breakdown,model_result = final_test(model, ftstldrs[i])
            results_file.write(f' Test accuracy: {round(model_result, 2)}')
            test_results.append(model_result)
            print(round(model_result, 1))
            test_breakdown = add_breakdowns(test_breakdown, model_breakdown)
        
    final_result = sum(test_results)/len(test_results)
    final_breakdown_result = avg_breakdowns(test_breakdown)
    results_file.write(f"\n\nArchitecture trained and tested {args.nummodl*args.kfolds} times on {args.setdir}, with an average accuracy of {round(final_result, 2)}.")
    results_file.write(f'\nTraining completed in {round((time.time()-prog_start)/60, 2)} minutes.')
    results_file.write(f'\n\nBreakdown of accuracy over problem types:')
    for ptype,acc in sorted(final_breakdown_result, key=lambda x : x[-1], reverse=True):
        results_file.write(f'\n    - {ptype:40s}\t{acc:>5.2f}')
    results_file.close()

if __name__ == '__main__':
    main()
    
    
# END SCRIPT -------------------------------------------------------------------------------------------------------- #