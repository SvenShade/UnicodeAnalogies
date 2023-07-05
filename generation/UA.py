# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Steven Spratley
# Date:    17/06/2022
# Purpose: Imports annotated images and generates the Unicode Analogies dataset


# IMPORTS ------------------------------------------------------------------------------------------------------------#


import os, random, shutil, numpy, time, cv2, csv
import matplotlib.pyplot as     plt
from   tqdm              import tqdm
from   matplotlib.image  import imread
from   collections       import defaultdict
from   itertools         import combinations


# CONSTANTS ----------------------------------------------------------------------------------------------------------#


ANNO_DIR = './Annotations'
SPLT_DIR = './Dataset splits'
RAND_SED = 1     # Random seed.
IMG_SIZE = 80    # Height/width of images in problems.
TRN_SPLT = 0.7   # Split between train and test problems (and images, if HOLD_OUT).
V_T_SPLT = 0.1   # Within test problems, split between validation and test.
SET_SIZE = 10000 # Number of problems to generate.
BINARISE = True  # Whether to output problems comprised of binary images (vs grayscale).
CON_SHFT = False # Whether to also generate shifted rule / conshift variants of problem types in val/tst.
HOLD_OUT = 'D'   # Hold-out controls which images are used in the construction of training and test sets.
                 # 'N' is none, 'B' is basic (train and test problems have shared images, but never presented with the
                 # same salient features), 'D' is difference (no set intersection between train and test images).
                 # 'D' results in much smaller datasets, if used with hard cull diversity (below).
MAX_DIVR = 'H'   # Whether to scale probabilities of rule-class instantiations such that diversity is maximised.
                 # 'N' is none, 'S' is scale probabilities for sampling, 'H' is hard cull.
                 # 'N' and 'S' guarantee datasets with SET_SIZE problems, but don't guarantee diversity like 'H'.
EXTRPLTE = 'N' if HOLD_OUT != 'N' else 'V'
                 # Extrapolation controls which rules and classes are held out. 
                 # 'N' is none, 'V' is class values, 'E' is rule-class pairs, 'P' is entire classes.
                 # If not HOLD_OUT, 'V' is selected by default to ensure separation of problems across trn/val/tst.
K_FOLDCV = 5     # Number of folds to generate for k-fold cross-validation.
DIVR_CAP = 300   # Upper limit on recognised folder sizes used to scale probabilities (if MAX_DIVR).
                 # Without this, rule-class instantiations with giant folders will dominate the problem pool.
CON_FAIL = 500   # Number of consecutive failures to endure during problem generation before bailing.
                 # Useful if MAX_DIVR is 'H', meaning that the limits of dataset expressivity might be reached before
                 # all SET_SIZE problems are generated.
NUM_DISP = 0     # Number of PMPs to load and display from the newly-generated dataset.
RULES    = ['constant', 'dist3', 'progression', 'arithmetic', 'union'] # Rules featured in the dataset.
OPPOSITE = {'NW': 'SE', 'N': 'S', 'NE': 'SW', 'E': 'W',
            'SE': 'NW', 'S': 'N', 'SW': 'NE', 'W': 'E'}
rule     = RULES[:2].capitalize() if type(RULES)!=list else 'A' if len(RULES)==5 else 'M'
SAVE_DIR = f'{SPLT_DIR}/{rule}-{EXTRPLTE}-{HOLD_OUT}-{"S" if CON_SHFT else "N"}-{MAX_DIVR}'

# Dictionary for problem specification. Keys are classes, while values are dicts that define valid problem types.
#   Constant = 'select' calls for both context rows to select separate subclasses to contrast (e.g. 'tall vs wide'), 
#   while constant = 'negate' provides the additional option for problems to be themed on concept negation (e.g.
#   'spaces vs no spaces'). Adding rules to each class's dict will guide problem generation. If 'progression' or 
#   'arithmetic' are specified, they should be valued with a list providing the order of subclasses. If the subclasses
#   are integer or cardinal values, specify 'integer' or 'cardinal'. If 'dist3' or 'union' are specified, they should 
#   be valued with lists specifying applicable subclasses (or None, if all are applicable).
CLASSES  = {'aspect':                           {'constant': 'select',
                                                 'progression': ['tall', 'square', 'wide']},
            'base-contacts':                    {'constant': 'select',
                                                 'progression': 'integer'},
            'base-style':                       {'constant': 'select',
                                                 'dist3': None},
            'closed':                           {'constant': 'negate',
                                                 'dist3': ['empty', 'halfshaded', 'full']},
            'closure':                          {'constant': 'negate',
                                                 'dist3': ['line', 'circle', 'square', 'triangle']},
            'components-internalsolid':         {'constant': 'negate',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'components-solid':                 {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'components-space':                 {'constant': 'negate',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'components-uniquesolid':           {'constant': 'select',
                                                 'progression': 'integer'},
            'concentricity':                    {'constant': 'select'},
            'connected-direction':              {'constant': 'negate'},
            'connected-quantity':               {'constant': 'negate',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'disconnected':                     {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'elongation':                       {'constant': 'select'},
            'gestalt':                          {'constant': 'select',
                                                 'union': None},
            'global-mass':                      {'constant': 'select',
                                                 'progression': 'cardinal'},
            'global-size':                      {'constant': 'select'},
            'group':                            {'constant': 'select'},
            'horns':                            {'constant': 'negate'},
            'ink':                              {'constant': 'select'},
            'interaction':                      {'constant': 'negate',
                                                 'progression': ['none', 'touching', 'overlap'],
                                                 'union': None},
            'intersection-angle':               {'constant': 'select',
                                                 'progression': ['acute', 'orthogonal', 'obtuse'],
                                                 'union': None},
            'intersection-emanating':           {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic': 'integer',
                                                 'union': None},
            'intersection-minimum':             {'constant': 'select',
                                                 'union': None},
            'intersection-quantity':            {'constant': 'negate',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'latin-style':                      {'constant':  'select',
                                                 'dist3':      None,
                                                 'union':  None},
            'latin-upper':                      {'constant': 'select',
                                                 'dist3':      None,
                                                 'union':  None},
            'negative':                         {'constant': 'negate'},
            'oddoneout':                        {'constant': 'negate'},
            'opening':                          {'constant': 'negate',
                                                 'union': None},
            'relational-position':              {'constant': 'select',
                                                 'dist3':     None,
                                                 'union': None},
            'relational-rotation':              {'constant': 'select',
                                                 'dist3':     None,
                                                 'union':     None},
            'relational-size':                  {'constant': 'select'},
            'shape-type':                       {'constant': 'select',
                                                 'union': None},
            'shape-sides':                      {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'stroke-arrow-quantity':            {'constant': 'select',
                                                 'progression': 'integer'},
            'stroke-dash-quantity':             {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'stroke-dot-quantity':              {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'stroke-feature':                   {'constant': 'select',
                                                 'dist3':      None,
                                                 'union':  None},
            'stroke-line-curve-quantity':       {'constant': 'select',
                                                 'progression': 'integer'},
            'stroke-line-loop-quantity':        {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'stroke-line-straight-quantity':    {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'stroke-line-u-quantity':           {'constant': 'select',
                                                 'progression': 'integer'},
            'stroke-line-wiggle-quantity':      {'constant': 'select',
                                                 'progression': 'integer'},
            'stroke-line-zig-quantity':         {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'stroke-style':                     {'constant': 'select',
                                                 'dist3': None,
                                                 'union': None},
            'symmetry-angle':                   {'constant': 'negate',
                                                 'progression': ['horizontal', 'diagonal', 'vertical'],
                                                 'union': None},
            'symmetry-rotational':              {'constant': 'negate',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'}}

GLOBAL_C = {'aspect':                           {'constant': 'select',
                                                 'progression': ['tall', 'square', 'wide']},
            'closure':                          {'constant': 'negate',
                                                 'dist3': ['line', 'circle', 'square', 'triangle']},
            'elongation':                       {'constant': 'select'},
            'gestalt':                          {'constant': 'select',
                                                 'union': None},
            'global-mass':                      {'constant': 'select',
                                                 'progression': 'cardinal'},
            'global-size':                      {'constant': 'select'},
            'ink':                              {'constant': 'select'},
            'negative':                         {'constant': 'negate'},
            'symmetry-angle':                   {'constant': 'negate',
                                                 'progression': ['horizontal', 'diagonal', 'vertical'],
                                                 'union': None},
            'symmetry-rotational':              {'constant': 'negate',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'}}

LOCAL_C  = {'base-contacts':                    {'constant': 'select',
                                                 'progression': 'integer'},
            'base-style':                       {'constant': 'select',
                                                 'dist3': None},
            'closed':                           {'constant': 'negate',
                                                 'dist3': ['empty', 'halfshaded', 'full']},
            'horns':                            {'constant': 'negate'},
            'intersection-angle':               {'constant': 'select',
                                                 'progression': ['acute', 'orthogonal', 'obtuse'],
                                                 'union': None},
            'intersection-emanating':           {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic': 'integer',
                                                 'union': None},
            'intersection-minimum':             {'constant': 'select',
                                                 'union': None},
            'intersection-quantity':            {'constant': 'negate',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'latin-style':                      {'constant':  'select',
                                                 'dist3':      None,
                                                 'union':  None},
            'latin-upper':                      {'constant': 'select',
                                                 'dist3':      None,
                                                 'union':  None},
            'opening':                          {'constant': 'negate',
                                                 'union': None},
            'shape-type':                       {'constant': 'select',
                                                 'union': None},
            'shape-sides':                      {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'stroke-arrow-quantity':            {'constant': 'select',
                                                 'progression': 'integer'},
            'stroke-dash-quantity':             {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'stroke-dot-quantity':              {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'stroke-feature':                   {'constant': 'select',
                                                 'dist3':      None,
                                                 'union':  None},
            'stroke-line-curve-quantity':       {'constant': 'select',
                                                 'progression': 'integer'},
            'stroke-line-loop-quantity':        {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'stroke-line-straight-quantity':    {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'stroke-line-u-quantity':           {'constant': 'select',
                                                 'progression': 'integer'},
            'stroke-line-wiggle-quantity':      {'constant': 'select',
                                                 'progression': 'integer'},
            'stroke-line-zig-quantity':         {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'stroke-style':                     {'constant': 'select',
                                                 'dist3': None,
                                                 'union': None}}

OBJREL_C = {'components-internalsolid':         {'constant': 'negate',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'components-solid':                 {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'components-space':                 {'constant': 'negate',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'components-uniquesolid':           {'constant': 'select',
                                                 'progression': 'integer'},
            'concentricity':                    {'constant': 'select'},
            'connected-direction':              {'constant': 'negate'},
            'connected-quantity':               {'constant': 'negate',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'disconnected':                     {'constant': 'select',
                                                 'progression': 'integer',
                                                 'arithmetic':  'integer'},
            'group':                            {'constant': 'select'},
            'interaction':                      {'constant': 'negate',
                                                 'progression': ['none', 'touching', 'overlap'],
                                                 'union': None},
            'oddoneout':                        {'constant': 'negate'},
            'relational-position':              {'constant': 'select',
                                                 'dist3':     None,
                                                 'union': None},
            'relational-rotation':              {'constant': 'select',
                                                 'dist3':     None,
                                                 'union':     None},
            'relational-size':                  {'constant': 'select'}}


# FUNCTIONS ----------------------------------------------------------------------------------------------------------#


# Read in image filenames from annotation folders and print statistics.
def collate_images():
    image_data = {}
    image_dict = defaultdict(int)
    print('\n\nBeginning import...')
    anno_folders = os.listdir(ANNO_DIR)
    print(f'  - Importing images from {len(anno_folders)} folders.')
    for folder in anno_folders:
        image_data[folder] = set(os.listdir(f'{ANNO_DIR}/{folder}'))
        for image_file in image_data[folder]:
            image_dict[image_file] += 1
    avg = round(sum(image_dict.values()) / float(len(image_dict.values())), 1)
    hi  = max(image_dict.values())
    print(f'  - {sum(image_dict.values())} annotations imported across {len(image_dict.keys())} unique images.')
    print(f'  - Images found possess {avg} annotations on average,')
    print(f'    with the most polysemic image possessing {hi}.')
    return image_data

# Load images. Using a separate function to scale and converte each image as loaded greatly minimises memory usage.
def load_img(fname):
    # Load and scale image.
    img = cv2.resize(imread(fname), dsize=(IMG_SIZE, IMG_SIZE))
    # Return grayscale image as binary white on black, if requested.
    return 1 - numpy.array(img, dtype=numpy.uint8 if BINARISE else numpy.float32)
    
# Places cardinal directions in a graph and selects 3 by taking a random walk.
# Option to return a run that makes a straight line (e.g. North, Centre, South).
def generate_cardinal_run(run_shape = 'straight'):
    cardinal_graph = {'C':  ['NW', 'N',  'NE', 'E', 'SE', 'S', 'SW', 'W'],
                      'NW': ['N',  'W',  'C'],
                      'N':  ['NW', 'NE', 'C'],
                      'NE': ['N',  'E',  'C'],
                      'E':  ['NE', 'SE', 'C'],
                      'SE': ['E',  'S',  'C'],
                      'S':  ['SW', 'SE', 'C'],
                      'SW': ['W',  'S',  'C'],
                      'W':  ['NW', 'SW', 'C']}
    straight_walks = [['NW', 'N', 'NE'], ['W', 'C', 'E'], ['SW', 'S', 'SE'],
                      ['NW', 'W', 'SW'], ['N', 'C', 'S'], ['NE', 'E', 'SE'],
                      ['NW', 'C', 'SE'], ['SW', 'C', 'NE']]
    if run_shape == 'straight':
        walk = random.choice(straight_walks)
        if random.choice([True, False]):
            walk.reverse()
        return walk
    walk = []
    directions = list(cardinal_graph.keys())
    for i in range(3): # Walk 3 steps through graph, at random.
        directions = list(set(directions).difference(set(walk))) # Don't revisit nodes.
        d = random.choice(directions)
        walk.append(d)
        directions = cardinal_graph[d]
    return walk

# Returns a sequence of three integers, at random from a given set of integers.
# A sequence is valid if it is in ascending or descending order, by increments of 1 or 2 (but not both).
def generate_integer_run(integers):
    possible_runs = []
    for i in integers:
        if i+1 in integers and i+2 in integers:
            possible_runs.append([i, i+1, i+2])
        if i+2 in integers and i+4 in integers:
            possible_runs.append([i, i+2, i+4])
    return sorted(random.choice(possible_runs), reverse=random.choice([True, False]))

# Returns a sequence of three integers, at random from a given set of integers.
# A sequence is valid if the third integer is the addition or difference of the prior two, and if the run is not
# already able to be generated by generate_integer_run (to avoid crossover of progression and arithmetic problems).
def generate_arithmetic_run(integers):
    possible_runs = []
    for i in integers:
        for j in integers:
            if i+j in integers and i-j != j-(i+j):
                possible_runs.append([i, j, i+j])
            if i-j in integers and i-j != j-(i-j):
                possible_runs.append([i, j, i-j])
    return random.choice(possible_runs)

def generate_random_constant_problem(image_data, clss=None, conshift=False):
    problem   = {'context': [], 'answers': [], 'label': None, 'rule': 'constant', 'instantiation': []}
    clss,data = random.choice(list(CLASSES.items())) if clss == None else (clss, CLASSES[clss])
    img_dirs  = [d for d in os.listdir(ANNO_DIR) if d[:len(clss)] == clss] if data['constant'] != 'select' else \
                [d for d in os.listdir(ANNO_DIR) if d[:len(clss)] == clss and d[-4:] != 'none']
    while True: #Ensure both features aren't 'none'.
        dir1 = random.choice(img_dirs)
        #If dir1 is a cardinal feature, and conshift is requested, pick dir2 to be the opposite direction.
        #This is under the intuition that problems with same or opposing directions are more intuitive.
        if dir1.split('-')[-1] in OPPOSITE and conshift:
            dir2 = dir1[:-len(dir1.split('-')[-1])] + OPPOSITE[dir1.split('-')[-1]]
        else:
            dir2 = random.choice(list(set(img_dirs).difference([dir1]))) if conshift else dir1
        if dir1[-4:]=='none' and dir2[-4:]=='none':
            continue
        break
    
    img_dirs.remove(dir2)
    problem['instantiation'] = [dir1, dir2] if dir1 != dir2 else [dir1]
    ims1,ims2 = image_data[dir1],image_data[dir2]
    if dir1 != dir2:
        ims1,ims2 = ims1.difference(ims2), ims2.difference(ims1)

    if len(ims1) >= 3 and len(ims2) >= 3 and len(ims1.union(ims2)) >= 6:
        if dir1 != dir2:
            row1,row2 = random.sample(ims1, 3),random.sample(ims2, 3)
        else:
            rows = random.sample(ims1, 6)
            row1,row2 = rows[:3],rows[3:]
    else:
        return None

    #Balance foil classes when sampling by only accepting at max 3 instances of each in the foil pool (ims3).
    ims3_lookup = {}
    invalid_foils = image_data[dir2].union(set(row1))
    for d in img_dirs:
        valid_foils = image_data[d].difference(invalid_foils)
        for f in random.sample(valid_foils, min(3, len(valid_foils))):
            ims3_lookup[f] = d
    ims3 = set(ims3_lookup.keys())
    if len(ims3) < 3:
        return None
    
    problem['context'] = [load_img(f'{ANNO_DIR}/{dir1}/{f}') for f in row1] + \
                         [load_img(f'{ANNO_DIR}/{dir2}/{f}') for f in row2]
    problem['answers'] = [problem['context'].pop(-1)] + \
                         [load_img(f'{ANNO_DIR}/{ims3_lookup[f]}/{f}') for f in random.sample(ims3, 3)]
    idxs = random.sample(range(4), 4)
    problem['label'] = idxs.index(0)
    problem['answers'] = [problem['answers'][i] for i in idxs]
    problem['answer_filename'] = row2[-1]
    problem['conshift'] = conshift
    problem['class'] = clss
    return problem

def generate_random_dist3_problem(image_data, clss=None, conshift=False):
    problem   = {'context': [], 'answers': [], 'label': None, 'rule': 'dist3', 'instantiation': []}
    classes   = [(k,v) for (k,v) in CLASSES.items() if 'dist3' in v]
    clss,data = random.choice(classes) if clss == None else (clss, CLASSES[clss])
    img_dirs  = [f'{clss}-{c}' for c in data['dist3']] if data['dist3'] \
                else [d for d in os.listdir(ANNO_DIR) if d[:len(clss)] == clss]
    #Choose random selection of three classes.
    dir1,dir2,dir3 = random.sample(img_dirs, 3)
    img_dirs.remove(dir3)
    problem['instantiation'] = [dir1, dir2, dir3]
    ims1,ims2,ims3 = image_data[dir1],image_data[dir2],image_data[dir3]

    if len(ims1) >= 2 and len(ims2) >= 2 and len(ims3) >= 2:
        context   = [random.sample(ims1, 2), random.sample(ims2, 2), random.sample(ims3, 2)]
        row1,row2 = [context[0][0], context[1][0], context[2][0]], [context[0][1], context[1][1], context[2][1]]
    else:
        return None

    #Balance foil classes when sampling by only accepting at max 3 instances of each in the foil pool (ims4).
    ims4_lookup = {}
    invalid_foils = image_data[dir3].union(set(row1 + row2))
    for d in img_dirs:
        valid_foils = image_data[d].difference(invalid_foils)
        for f in random.sample(valid_foils, min(3, len(valid_foils))):
            ims4_lookup[f] = d
    ims4 = set(ims4_lookup.keys())
    if len(ims4) < 3:
        return None
    
    problem['context'] = [load_img(f'{ANNO_DIR}/{dir1}/{row1[0]}'),
                          load_img(f'{ANNO_DIR}/{dir2}/{row1[1]}'),
                          load_img(f'{ANNO_DIR}/{dir3}/{row1[2]}')] + \
                         [load_img(f'{ANNO_DIR}/{dir1}/{row2[0]}'),
                          load_img(f'{ANNO_DIR}/{dir2}/{row2[1]}'),
                          load_img(f'{ANNO_DIR}/{dir3}/{row2[2]}')]
    problem['answers'] = [problem['context'].pop(-1)] + \
                         [load_img(f'{ANNO_DIR}/{ims4_lookup[f]}/{f}') for f in random.sample(ims4, 3)]
    if conshift:
        problem['context'] = random.sample(problem['context'][:3], 3) + problem['context'][3:]
    idxs = random.sample(range(4), 4)
    problem['label'] = idxs.index(0)
    problem['answers'] = [problem['answers'][i] for i in idxs]
    problem['answer_filename'] = row2[-1]
    problem['conshift'] = conshift
    problem['class'] = clss
    return problem

def generate_random_progression_problem(image_data, clss=None, conshift=False):
    problem   = {'context': [], 'answers': [], 'label': None, 'rule': 'progression', 'instantiation': []}
    classes   = [(k,v) for (k,v) in CLASSES.items() if 'progression' in v]
    clss,data = random.choice(classes) if clss == None else (clss, CLASSES[clss])
    #Progression can be integer, cardinal, or other (provided as a list)
    img_dirs = [d for d in os.listdir(ANNO_DIR) if d[:len(clss)] == clss]
    #Choose a random progression sequence of three classes (or two different progressions, if conshift)
    while True:
        for i in range(2 if conshift else 1):
            if data['progression'] == 'integer':
                img_dirs   = [d for d in img_dirs if d[-4:] != 'none']
                clss_order = sorted(img_dirs, key = lambda d : int(d.split('-')[-1]))
                integers   = [int(d.split('-')[-1]) for d in clss_order]
                int_idxs   = {i:c for i,c in zip(integers, clss_order)}
                clss_order = generate_integer_run(integers)
                dir1,dir2,dir3 = int_idxs[clss_order[0]],int_idxs[clss_order[1]],int_idxs[clss_order[2]]
            elif data['progression'] == 'cardinal':
                dir1,dir2,dir3 = [f'{clss}-{c}' for c in generate_cardinal_run()]
            else:
                clsses         = [f'{clss}-{c}' for c in data['progression']]
                clss_order     = clsses if random.choice([True, False]) else list(reversed(clsses))
                seq_start      = random.choice(clss_order[:-2])
                dir1,dir2,dir3 = clss_order[clss_order.index(seq_start) : clss_order.index(seq_start)+3]
            if i==0:
                dir4,dir5,dir6 = dir1,dir2,dir3
        #If conshift requested, but both rows have the exact sequence by chance, reroll until they're different.
        if conshift and dir4+dir5+dir6 == dir1+dir2+dir3:
            continue
        else:
            break

    ims1,ims2,ims3 = image_data[dir1],image_data[dir2],image_data[dir3]
    ims1,ims2,ims3 = ims1.difference(ims2.union(ims3)),\
                     ims2.difference(ims1.union(ims3)),\
                     ims3.difference(ims1.union(ims2))

    if conshift:
        problem['instantiation'] = [dir1, dir2, dir3, dir4, dir5, dir6]
        ims4,ims5,ims6 = image_data[dir4],image_data[dir5],image_data[dir6]
        ims4,ims5,ims6 = ims4.difference(ims5.union(ims6)),\
                         ims5.difference(ims4.union(ims6)),\
                         ims6.difference(ims4.union(ims5))
        try:
            row1 = [random.choice(list(ims1)), random.choice(list(ims2)), random.choice(list(ims3))]
            row2 = [random.choice(list(ims4)), random.choice(list(ims5)), random.choice(list(ims6))]
        except IndexError:
            return None
    else:
        problem['instantiation'] = [dir1, dir2, dir3]
        if len(ims1) >= 2 and len(ims2) >= 2 and len(ims3) >= 2:
            context   = [random.sample(ims1, 2), random.sample(ims2, 2), random.sample(ims3, 2)]
            row1,row2 = [context[0][0], context[1][0], context[2][0]], [context[0][1], context[1][1], context[2][1]]
        else:
            return None

    #Balance foil classes when sampling by only accepting at max 3 instances of each in the foil pool (ims7).
    ims7_lookup = {}
    invalid_foils = image_data[dir6 if conshift else dir3].union(set(row1 + row2))
    for d in img_dirs:
        valid_foils = image_data[d].difference(invalid_foils)
        for f in random.sample(valid_foils, min(3, len(valid_foils))):
            ims7_lookup[f] = d
    ims7 = set(ims7_lookup.keys())
    if len(ims7) < 3:
        return None
    
    problem['context'] = [load_img(f'{ANNO_DIR}/{dir1}/{row1[0]}'),
                          load_img(f'{ANNO_DIR}/{dir2}/{row1[1]}'),
                          load_img(f'{ANNO_DIR}/{dir3}/{row1[2]}')] + \
                         [load_img(f'{ANNO_DIR}/{dir4 if conshift else dir1}/{row2[0]}'),
                          load_img(f'{ANNO_DIR}/{dir5 if conshift else dir2}/{row2[1]}'),
                          load_img(f'{ANNO_DIR}/{dir6 if conshift else dir3}/{row2[2]}')]
    problem['answers'] = [problem['context'].pop(-1)] + \
                         [load_img(f'{ANNO_DIR}/{ims7_lookup[f]}/{f}') for f in random.sample(ims7, 3)]
    idxs = random.sample(range(4), 4)
    problem['label'] = idxs.index(0)
    problem['answers'] = [problem['answers'][i] for i in idxs]
    problem['answer_filename'] = row2[-1]
    problem['conshift'] = conshift
    problem['class'] = clss
    return problem

def generate_random_arithmetic_problem(image_data, clss=None, conshift=False):
    problem   = {'context': [], 'answers': [], 'label': None, 'rule': 'arithmetic', 'instantiation': []}
    classes   = [(k,v) for (k,v) in CLASSES.items() if 'arithmetic' in v]
    clss,data = random.choice(classes) if clss == None else (clss, CLASSES[clss])
    img_dirs = [d for d in os.listdir(ANNO_DIR) if d[:len(clss)] == clss and d[-4:] != 'none']
    #Choose random arithmetic sequence of three integer classes.
    while True:
        for i in range(2 if conshift else 1):
            img_dirs   = [d for d in img_dirs if d[-4:] != 'none']
            clss_order = sorted(img_dirs, key = lambda d : int(d.split('-')[-1]))
            integers   = [int(d.split('-')[-1]) for d in clss_order]
            int_idxs   = {i:c for i,c in zip(integers, clss_order)}
            j,k,l      = generate_arithmetic_run(integers)
            alt_ans_1  = (j+k) if (j+k != l) else (j-k)
            dir1,dir2,dir3 = int_idxs[j],int_idxs[k],int_idxs[l]
            if i==0:
                dir4,dir5,dir6 = dir1,dir2,dir3
                alt_ans_2 = alt_ans_1
        #If conshift requested, but both rows have the exact sequence by chance, reroll until they're different.
        if conshift and dir4+dir5+dir6 == dir1+dir2+dir3:
            continue
        else:
            break
    
    ims1,ims2,ims3 = image_data[dir1],image_data[dir2],image_data[dir3]
    if conshift:
        problem['instantiation'] = [dir1, dir2, dir3, dir4, dir5, dir6]
        ims4,ims5,ims6 = set(image_data[dir4]),\
                         set(image_data[dir5]),\
                         set(image_data[dir6])
        try:
            row1 = [random.choice(list(ims1)), random.choice(list(ims2)), random.choice(list(ims3))]
            row2 = [random.choice(list(ims4)), random.choice(list(ims5)), random.choice(list(ims6))]
        except IndexError:
            return None
    else:
        problem['instantiation'] = [dir1, dir2, dir3]
        if len(ims1) >= 2 and len(ims2) >= 2 and len(ims3) >= 2:
            context   = [random.sample(ims1, 2), random.sample(ims2, 2), random.sample(ims3, 2)]
            row1,row2 = [context[0][0], context[1][0], context[2][0]], [context[0][1], context[1][1], context[2][1]]
        else:
            return None

    #Balance foil classes when sampling by only accepting at max 3 instances of each in the foil pool (ims7).
    ims7_lookup = {}
    if conshift:
        #Ensure that foils pertaining to alternative answers (an arithmetic sequence could be + or -) are gone.
        invalid_foils = image_data[dir6].union(set(row1 + row2))
        if alt_ans_2 in int_idxs:
            invalid_foils = invalid_foils.union(image_data[int_idxs[alt_ans_2]])
    else:
        invalid_foils = image_data[dir3].union(set(row1 + row2))
    for d in img_dirs:
        valid_foils = image_data[d].difference(invalid_foils)
        for f in random.sample(valid_foils, min(3, len(valid_foils))):
            ims7_lookup[f] = d
    ims7 = set(ims7_lookup.keys())
    if len(ims7) < 3:
        return None
    
    problem['context'] = [load_img(f'{ANNO_DIR}/{dir1}/{row1[0]}'),
                          load_img(f'{ANNO_DIR}/{dir2}/{row1[1]}'),
                          load_img(f'{ANNO_DIR}/{dir3}/{row1[2]}')] + \
                         [load_img(f'{ANNO_DIR}/{dir4 if conshift else dir1}/{row2[0]}'),
                          load_img(f'{ANNO_DIR}/{dir5 if conshift else dir2}/{row2[1]}'),
                          load_img(f'{ANNO_DIR}/{dir6 if conshift else dir3}/{row2[2]}')]
    problem['answers'] = [problem['context'].pop(-1)] + \
                         [load_img(f'{ANNO_DIR}/{ims7_lookup[f]}/{f}') for f in random.sample(ims7, 3)]
    idxs = random.sample(range(4), 4)
    problem['label'] = idxs.index(0)
    problem['answers'] = [problem['answers'][i] for i in idxs]
    problem['answer_filename'] = row2[-1]
    problem['conshift'] = conshift
    problem['class'] = clss
    return problem

def generate_random_union_problem(image_data, clss=None, conshift=False):
    problem   = {'context': [], 'answers': [], 'label': None, 'rule': 'union', 'instantiation': []}
    classes   = [(k,v) for (k,v) in CLASSES.items() if 'union' in v]
    clss,data = random.choice(classes) if clss == None else (clss, CLASSES[clss])
    img_dirs  = [f'{clss}-{c}' for c in data['union']] if data['union'] \
                else [d for d in os.listdir(ANNO_DIR) if d[:len(clss)] == clss]
    #Find features that can be used for union problems by taking the set intersections between image directories.
    d_dict = {}
    for d1,d2 in combinations(img_dirs, 2):
        d1_imgs,d2_imgs = image_data[d1],image_data[d2]
        d_dict[(d1, d2)] = d1_imgs.intersection(d2_imgs)
    d_dict = {k:v for (k,v) in d_dict.items() if (len(v) > 1 if not conshift else len(v) >= 1)}
    
    if (not d_dict) or (conshift and len(d_dict)<2):
        return None
    else:
        # Select the first five context images.
        if conshift:
            # Pick union images (ims 3 and 6)
            ((d1,d2),ims3),((d4,d5),ims6) = random.sample(list(d_dict.items()), 2)
            # Given union images, grab sets of images to make up the rest of the problem.
            # Ensure that these images do not possess both union features by performing a set difference.
            ims1,ims2,ims4,ims5 = image_data[d1],image_data[d2],image_data[d4],image_data[d5]
            ims1,ims2 = ims1.difference(ims2),ims2.difference(ims1)
            ims4,ims5 = ims4.difference(ims5),ims5.difference(ims4)
            # Try selecting particular images to populate the problem. Bail if ever a random choice is to be made 
            # from an empty sequence (i.e. not enough images to instantiate this particular problem).
            try:
                # Select at random, the first five context images.
                img1,img2,img3,img4,img5 = random.choice(list(ims1)),random.choice(list(ims2)),random.choice(list(ims3)),\
                                           random.choice(list(ims4)),random.choice(list(ims5))
            except IndexError:
                return None
        else:
            # Pick union images (ims3) and perform set difference for first two image sets (ims1 and ims2)
            ((d1,d2),ims3) = random.choice(list(d_dict.items()))
            d4,d5 = d1,d2
            ims1,ims2 = image_data[d1],image_data[d2]
            ims1,ims2 = ims1.difference(ims2),ims2.difference(ims1)
            # Try selecting images.
            try:
                (img1,img4),(img2,img5),img3 = random.sample(list(ims1), 2),random.sample(list(ims2), 2),\
                                               random.choice(list(ims3))
            except:
                return None
        
        # Check for any other valid union rules present in the second row.
        # Prohibit images that would accidentally permit a second rule forming.            
        second_rule = set()
        all_images = []
        for d in img_dirs:
            all_images.extend(image_data[d])
        for d_1 in [d for d in img_dirs if img4 in image_data[d]]:
            for d_2 in [d for d in img_dirs if img5 in image_data[d] and d != d_1]:
                if (d4,d5) not in [(d_1,d_2), (d_2,d_1)]:
                    for img in all_images:
                        if img in image_data[d_1] and img in image_data[d_2]:
                            second_rule.add(img)
        
        # Select the final context image. Ensure it's not the same as img3.
        ims6 = ims6.difference(second_rule) if conshift else ims3.difference(second_rule)
        ims6 = list(ims6.difference(set([img3])))
        if ims6:
            img6 = random.choice(ims6)
        else:
            return None
    
    #Initialise invalid foils set. Add images that:
    # - would accidentally complete the sequence to the intended rule (ims6 or ims3)
    # - would accidentally complete the sequence to an emergent rule (second_rule)
    # - have already been selected in the problem context
    row1,row2 = [img1,img2,img3],[img4,img5,img6]
    invalid_foils = set(list(ims6 if conshift else ims3) + row1 + row2 + list(second_rule))
    #Balance foil classes when sampling by only accepting at max 3 instances of each in the foil pool (ims7).
    ims7_lookup   = {}
    problem['instantiation'] = [d1, d2, d4, d5] if conshift else [d1, d2]
    for d in problem['instantiation']:
        valid_foils = image_data[d].difference(invalid_foils)
        for f in random.sample(valid_foils, min(3, len(valid_foils))):
            ims7_lookup[f] = d
    ims7 = set(ims7_lookup.keys())
    if len(ims7) < 3:
        return None
    
    problem['context'] = [load_img(f'{ANNO_DIR}/{d1}/{row1[0]}'),
                          load_img(f'{ANNO_DIR}/{d2}/{row1[1]}'),
                          load_img(f'{ANNO_DIR}/{d2}/{row1[2]}')] + \
                         [load_img(f'{ANNO_DIR}/{d4}/{row2[0]}'),
                          load_img(f'{ANNO_DIR}/{d5}/{row2[1]}'),
                          load_img(f'{ANNO_DIR}/{d5}/{row2[2]}')]
    problem['answers'] = [problem['context'].pop(-1)] + \
                         [load_img(f'{ANNO_DIR}/{ims7_lookup[f]}/{f}') for f in random.sample(ims7, 3)]
    idxs = random.sample(range(4), 4)
    problem['label'] = idxs.index(0)
    problem['answers'] = [problem['answers'][i] for i in idxs]
    problem['answer_filename'] = row2[-1]
    problem['conshift'] = conshift
    problem['class'] = clss
    return problem

# Generates set_size number of problems.
# If weighted_sampling, samples problem rules with probabilities proportional to the number of images applicable to those rules.
# In doing so, dataset becomes unbalanced regarding rule representation, but diversity of dataset is maximised.
# If not weighted_sampling, dataset maintains equal representation of all rules, but risks encouraging memorisation of
# rule types that have less images available for their construction.
# If MAX_DIVR is 'H', balance the dataset by ensuring there exists no more than one problem of any rule-class 
# instantiation that shares the same correct answer, creating a diverse dataset to mitigate model memorisation.
def generate_problems(image_data, prb_types, set_size, mode, weighted_sampling):
    problems,failures,consecutive_fails = [],set(),0
    dataset_dict = {}
    # Create 1D arrays from prb_types with which to define problem sampling.
    types   = numpy.asarray([f'{r}_{c}_{h}' for [r,c,h,_] in prb_types])
    weights = numpy.asarray([t[-1] for t in prb_types] if weighted_sampling=='S' else
                            [1/len(prb_types) for _ in prb_types])
    with tqdm(total=set_size, desc=f'    - {mode.capitalize()} set\t') as pbar:
        while len(problems) < set_size and consecutive_fails < CON_FAIL:
            # Sample a problem type and try to instantiate it as a valid problem.
            sampled_type = numpy.random.choice(types, p=weights)
            rule,clss,conshift = sampled_type.split('_')
            # Reroll production up to 50 times if the last roll didn't successfully yield a problem.
            i,p = 0,None
            while not p and not i==50:
                p = globals()[f'generate_random_{rule}_problem'](image_data, clss=clss, conshift=conshift=='True')
                i += 1
            if p:
                ptype = f'{p["rule"]}_{p["instantiation"]}_{p["answer_filename"]}'
                if (MAX_DIVR=='H' and ptype not in dataset_dict) or (MAX_DIVR!='H'):
                    problems.append(p)
                    dataset_dict[ptype] = p
                    pbar.update(1)
                    consecutive_fails = 0
                else:
                    consecutive_fails += 1
            else:
                failures.add(f'{rule}-{clss}-{conshift}')
    if consecutive_fails >= CON_FAIL:
        print('\nWarning: Under requested parameters, limits of dataset expressivity reached before all requested problems generated.')
    return problems,failures

# Add conshift problem types to the list of types to be sampled from, as requested.
# Then, convert folder sizes associated with problem types into weights (i.e. relative sizes, given total images).
def add_conshift_find_weights(prb_types, mode):
    types = []
    for rule,clss,size in prb_types:
        for conshift in [True, False] if ((mode=='test' and CON_SHFT) or (mode=='train')) else [False]:
            types.append([rule,clss,conshift,size])
    total_images = sum([t[-1] for t in types])
    types = [[rule,clss,conshift,size/total_images] for [rule,clss,conshift,size] in types]
    return types

# Populates a new dataset with a number of problems (requested by SET_SIZE).
# Returns a (train, validation, test) tuple consisting of problem lists.
def generate_dataset(image_data, all_classes=CLASSES):
    print('\nGenerating dataset...')
    # prb_types holds all problem types relevant to the requested dataset. It also ennumerates the images available to
    # each problem type, which is used in scaling problem sampling for diversity.
    prb_types = []
    classes   = set()
    for rule in RULES if type(RULES)==list else [RULES]:
        for clss,details in all_classes.items():
            if rule in details:
                classes.add(clss)
                size = 0
                for d in os.listdir(ANNO_DIR):
                    if d[:len(clss)] == clss:
                        size += len(os.listdir(f'{ANNO_DIR}/{d}'))
                prb_types.append([rule,clss,min(size, DIVR_CAP)])
    # Partition problem types given requested level of extrapolation (EXTRPLTE).
    if EXTRPLTE == 'E':
        types    = random.sample(prb_types, len(prb_types))
        ex_split = round(len(types)*TRN_SPLT)
        trn_types,tst_types = types[:ex_split],types[ex_split:]
    elif EXTRPLTE == 'P':
        types    = random.sample(prb_types, len(prb_types))
        classes  = random.sample(classes, len(classes))
        ex_split = round(len(classes)*TRN_SPLT)
        trn_classes,tst_classes = classes[:ex_split],classes[ex_split:]
        trn_types,tst_types = [t for t in prb_types if t[1] in trn_classes],[t for t in prb_types if t[1] in tst_classes]
    else: # If no extrapolation requested, training and test problems sample the same rules and classes.
        trn_types = tst_types = prb_types
    # Add conshift problem types, then turn image numbers associated with types into weights, given the overall number 
    # of images available to that type.
    trn_types,tst_types = add_conshift_find_weights(trn_types, mode='train'),\
                          add_conshift_find_weights(tst_types, mode='test')
    total_num_types = len(trn_types+tst_types) if EXTRPLTE in ['E', 'P'] else len(trn_types)
    # Report rule types being instantiated.
    print(f'  - Instantiating {RULES} rules.')
    # Partition images given requested hold-out (HOLD_OUT).
    if HOLD_OUT != 'N':
        if HOLD_OUT == 'D':
            # Hold-out strategy: complete set difference. Leads to much smaller datasets.
            image_set = set()
            for folder in image_data.values():
                image_set = image_set.union(folder)
            unique_images = random.sample(list(image_set), len(image_set))
            split_point   = round(len(unique_images)*TRN_SPLT)
            trn_img_set,tst_img_set = set(unique_images[:split_point]),set(unique_images[split_point:])
            trn_data,tst_data = {},{}
            trn_anno,tst_anno = 0,0
            for folder,files in image_data.items():
                trn_data[folder],tst_data[folder] = trn_img_set.intersection(files),tst_img_set.intersection(files)
                trn_anno += len(trn_data[folder])
                tst_anno += len(tst_data[folder])
        else:
            # Hold-out strategy: basic. Allows common images across training and test, provided that they aren't used
            # for the same annotation data.
            trn_data,tst_data = {},{}
            trn_anno,tst_anno = 0,0
            trn_img_set,tst_img_set = set(),set()
            for f in image_data.keys():
                images    = random.sample(image_data[f], len(image_data[f]))
                im_split  = round(len(images)*TRN_SPLT)
                trn_data[f],tst_data[f] = set(images[:im_split]),set(images[im_split:])
                trn_anno += len(trn_data[f])
                tst_anno += len(tst_data[f])
                trn_img_set,tst_img_set = trn_img_set.union(set(images[:im_split])),tst_img_set.union(set(images[im_split:]))
            
        print(f'  - {"Set difference" if HOLD_OUT=="D" else "Basic"} hold-out requested, selecting:')
        print(f'    - {trn_anno} annotations and {len(trn_img_set)} unique images with which to form training problems,')
        print(f'    - {tst_anno} annotations and {len(tst_img_set)} unique images for val/test problems.')
        
    else:
        trn_data = tst_data = image_data
        print(f'  - No hold-out requested, so using all available images to form training and val/test problems.')
        
    # Report on other settings.
    if EXTRPLTE=='E':
        print(f'  - Extrapolation requested, sampling {len(trn_types)} problem types in train, and {len(tst_types)} in val/test.')
    elif EXTRPLTE=='P':
        print(f'  - "Extrapolation+" requested: partitioning classes and sampling {len(trn_types)} problem types in train, and {len(tst_types)} in val/test.')
    elif EXTRPLTE=='V':
        print(f'  - Value extrapolation requested: repartitioning sets to ensure problem types are instantiated differently across train/val/test.')
    else:
        print(f'  - Extrapolation not requested, sampling all problem types in train and test.')
        if HOLD_OUT=='N':
            print('    However, value extrapolation is turned on given no held-out images; repartitioning sets to ensure problem types')
            print('    are indeed instantiated differently across train/val/test such that there is no intersection.')
    if MAX_DIVR=='S':
        print(f'  - Maximum diversity by sampling requested, so generating {SET_SIZE} problems by weighted sampling of {total_num_types} problem types.')
    elif MAX_DIVR=='H':
        print(f'  - Maximum diversity by hard cull requested; even sampling {total_num_types} problem types while disallowing problems with the same')
        print(f'    rule-class tuples from sharing correct answers, in order to discourage model memorisation.')
    else:
        print(f'  - Maximum diversity not requested, so generating {SET_SIZE} problems by even sampling of {total_num_types} problem types.')
    if CON_SHFT:
        print('  - Context shift requested.')
    else:
        print('  - Context shift not requested.')
    
    # Generate train set.
    trn_fail,tst_fail = set(),set()
    trn,trn_fail = generate_problems(trn_data, trn_types, round(SET_SIZE*TRN_SPLT), 'train', weighted_sampling=MAX_DIVR)
    # Generate validation and test sets.
    tst,tst_fail = generate_problems(tst_data, tst_types, round(SET_SIZE*(1-TRN_SPLT)), 'test', weighted_sampling=MAX_DIVR)
    val,tst = tst[:round(len(tst)*V_T_SPLT)],tst[round(len(tst)*V_T_SPLT):] # Split test set into test and validation sets.
    
    # If images aren't held out, repartition train/validation/test problems to ensure the same rule-class-value tuple
    # isn't instantiated across splits.
    if HOLD_OUT=='N':
        all_prob_types = defaultdict(list)
        for p in trn+val+tst:
            description = f'{p["rule"]}_{p["instantiation"]}'
            all_prob_types[description].append(p)
        pt = random.sample(list(all_prob_types.keys()), len(all_prob_types.keys()))
        trn_t,tst_t = pt[:round(len(pt)*TRN_SPLT)],pt[round(len(pt)*TRN_SPLT):]
        val_t,tst_t = tst_t[:round(len(tst_t)*V_T_SPLT)],tst_t[round(len(tst_t)*V_T_SPLT):]
        trn,val,tst = [],[],[]
        for pt in trn_t:
            trn.extend(all_prob_types[pt])
        for pt in val_t:
            val.extend(all_prob_types[pt])
        for pt in tst_t:
            tst.extend(all_prob_types[pt])
            
    print(f'\n  - Successfully generated {len(trn)} train problems.')
    print(f'  - Successfully generated {len(val)} validation and {len(tst)} test problems.')
    print(f'  - Failed to instantiate {len(trn_fail)}/{len(trn_types)} problem types in the train set.')
    print(f'  - Failed to instantiate {len(tst_fail)}/{len(tst_types)} problem types in the val+test set.\n')
    print(f'Problem types failed: {"" if (trn_fail or tst_fail) else "None."}')
    if trn_fail:
        print(f'  - Train:\n{trn_fail}\n')
    if tst_fail:
        print(f'  - Val/Test:\n{tst_fail}')
    return trn,val,tst

def make_border(img):
    w = img.shape[0]
    for i in range(w):
        img[0][i] = img[w-1][i] = img[i][0] = img[i][w-1] = 0
    return img

def print_problem(p, titled=True, borders=True, print_location=None):
    if not p:
        return None
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    if titled:
        inst = p["instantiation"]
        inst_str = inst[0] if len(inst)==1 \
                           else f'{inst[0]} and {inst[1]}' if len(inst)==2 \
                           else f'{inst[0]}, {inst[1]}, and {inst[2]}' if len(inst)==3 \
                           else f'{inst[0]}, {inst[1]} first, then {inst[2]}, {inst[3]}' if len(inst)==4 \
                           else f'{inst[0]}, {inst[1]}, and {inst[2]} first, \nthen {inst[3]}, {inst[4]}, and {inst[5]}'
        fig.suptitle(f'{p["rule"].capitalize()} in {inst_str}, answer={p["label"]+1}.', fontdict = {'fontsize' : 12}, y=-0.01)
    ax = axes.flatten()
    for i in range(12):
        ax[i].set_axis_off()
    for i in range(5):
        plot_pos = [0,1,2,6,7]
        img = p['context'][i]
        ax[plot_pos[i]].imshow(make_border(img) if borders else img, cmap="gray")
    for i in range(4):
        plot_pos = [4,5,10,11]
        img = p['answers'][i]
        ax[plot_pos[i]].imshow(make_border(img) if borders else img, cmap="gray")
    fig.tight_layout()
    if print_location:
        plt.savefig(print_location)
    else:
        plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

# Commit a dataset to disk.
def save_dataset(dataset, path=SAVE_DIR, img_size=IMG_SIZE):
    print('\nSaving new dataset to disk...')
    if os.path.exists(path): # Delete a previously generated dataset, if exists.
        print('  - Deleting existing dataset directory and contents.')
        shutil.rmtree(path)
    os.mkdir(path)
    occurrences = defaultdict(int)
    trn,val,tst = dataset
    for mode,probs in [('train',trn), ('val',val), ('test',tst)]:
        print(f'  - Saving {mode} set.')
        for p in probs:
            if p:
                description = f'{p["rule"]}_{p["instantiation"]}'
                occurrences[description] += 1
                instnt = ''
                for i in p['instantiation']:
                    instnt += f'_{i}'
                f_name = f'{path}/{mode}_{p["rule"]}{instnt}_{occurrences[description]+1}'
                images = p['context']+p['answers']
                target = p['label']
                probtype = f"{p['rule']}_{p['class']}_{'shift' if p['conshift'] else 'noshift'}"
                numpy.savez_compressed(f_name, images=images, target=target, probtype=probtype)
                    
# Load and display a random selection of problems from the saved dataset.
def observe_dataset(path=SAVE_DIR, num_display=NUM_DISP):
    if not num_display:
        return None
    prbfiles = [f for f in os.listdir(path) if f[-3:]=='npz']
    problems = [numpy.load(f'{SAVE_DIR}/{p}') for p in random.sample(prbfiles, num_display)]
    for p in problems:
        p = {'context': 1-p['images'][:5], 'answers': 1-p['images'][5:]}
        print_problem(p, titled=False, borders=True)

# Render images (jpgs) of all problems in the saved dataset.
def render_dataset():
    problems,rendered,left = [],[],[]
    for f in os.listdir(SAVE_DIR):
        if f[-3:]=='npz':
            problems.append(f[:-4])
        else:
            rendered.append(f[:-9])
    rendered = set(rendered)
    for f in problems:
        if f not in rendered:
            left.append(f+'.npz')
    for f in tqdm(left, desc='Renders'):
        with numpy.load(f'{SAVE_DIR}/{f}') as p:
            answer = p['target'] + 1
            print_problem({'context': 1-p['images'][:5], 'answers': 1-p['images'][5:]}, 
                          titled=False, borders=True, print_location=f'{SAVE_DIR}/{f[:-4]}_ans{answer}.jpg')
        
# Load dataset, and copy a number of problems per each problem type into another folder.
# Balanced set may be sampled by survey software in order for human participants to establish a baseline.
def balance_dataset(num_per_type=5, copy_img=False):
    # Remake directory.
    balanced_dir = f'{SAVE_DIR}_balanced'
    if os.path.exists(balanced_dir):
        shutil.rmtree(balanced_dir)
    os.mkdir(balanced_dir)
    os.mkdir(f'{balanced_dir}/all_probs')
    os.mkdir(f'{balanced_dir}/all_probs/train')
    os.mkdir(f'{balanced_dir}/all_probs/test')
    os.mkdir(f'{balanced_dir}/by_type')
    os.mkdir(f'{balanced_dir}/by_type/train')
    os.mkdir(f'{balanced_dir}/by_type/test')
    # Sample num_per_type problems at random, per train and test partition.
    trn_dict = defaultdict(list)
    tst_dict = defaultdict(list)
    problems = [f for f in os.listdir(SAVE_DIR) if f[-3:]=='npz']
    renders  = [f for f in os.listdir(SAVE_DIR) if f[-3:]=='jpg']
    problems = random.sample(problems, len(problems))
    trn_prbs = [f for f in problems if f.split('_')[0]=='train']
    tst_prbs = [f for f in problems if f.split('_')[0]=='test']
    for f in trn_prbs:
        with numpy.load(f'{SAVE_DIR}/{f}') as p:
            if len(trn_dict[str(p['probtype'])]) < num_per_type:
                trn_dict[str(p['probtype'])].append(f)
    for f in tst_prbs:
        with numpy.load(f'{SAVE_DIR}/{f}') as p:
            if len(tst_dict[str(p['probtype'])]) < num_per_type:
                tst_dict[str(p['probtype'])].append(f)
    for probtype,problist in trn_dict.items():
        for f in problist:
            path = f'{balanced_dir}/by_type/train/{probtype}'
            if not os.path.exists(path):
                os.mkdir(path)
            if copy_img:
                for r in renders:
                    if f[:-4]==r[:-9]:
                        f = r
            shutil.copyfile(f'{SAVE_DIR}/{f}', f'{path}/{f}')
            shutil.copyfile(f'{SAVE_DIR}/{f}', f'{balanced_dir}/all_probs/train/{f}')
    for probtype,problist in tst_dict.items():
        for f in problist:
            path = f'{balanced_dir}/by_type/test/{probtype}'
            if not os.path.exists(path):
                os.mkdir(path)
            if copy_img:
                for r in renders:
                    if f[:-4]==r[:-9]:
                        f = r
            shutil.copyfile(f'{SAVE_DIR}/{f}', f'{path}/{f}')
            shutil.copyfile(f'{SAVE_DIR}/{f}', f'{balanced_dir}/all_probs/test/{f}')

        
# MAIN ---------------------------------------------------------------------------------------------------------------#


start_time = time.time()

# Generate datasets featuring individual rules, k folds each (experiment #1 in paper).
rules_all = RULES
for r in rules_all:
    # Delete and remake save directory.
    RULES = [r]
    SAVE_DIR = f'{SPLT_DIR}/{r[:2].capitalize()}-{EXTRPLTE}-{HOLD_OUT}-{"S" if CON_SHFT else "N"}-{MAX_DIVR}'
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.mkdir(SAVE_DIR)
    # Generate and save k folds.
    for i in range(K_FOLDCV):
        # Increment random seeds per fold.
        random.seed(RAND_SED+i)
        numpy.random.seed(RAND_SED+i)
        # Re-collate and re-generate dataset per fold.
        image_data = collate_images()
        dataset    = generate_dataset(image_data)
        # Save fold to disk.
        save_dataset(dataset, path=f'{SAVE_DIR}/fold_{i}/')
print(f'\nExperiment 1 splits generated and saved in {round((time.time()-start_time)/60, 2)} minutes.')

# Generate datasets featuring individual schema categories, k folds each (experiment #2).
for name,categ in [('GLOB',GLOBAL_C), ('LOC',LOCAL_C), ('OBJ',OBJREL_C)]:
    # Delete and remake save directory.
    RULES = rules_all
    SAVE_DIR = f'{SPLT_DIR}/{name}-A-{EXTRPLTE}-{HOLD_OUT}-{"S" if CON_SHFT else "N"}-{MAX_DIVR}'
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.mkdir(SAVE_DIR)
    # Generate and save k folds.
    for i in range(K_FOLDCV):
        # Increment random seeds per fold.
        random.seed(RAND_SED+i)
        numpy.random.seed(RAND_SED+i)
        # Re-collate and re-generate dataset per fold.
        image_data = collate_images()
        dataset    = generate_dataset(image_data, all_classes=categ)
        # Save fold to disk.
        save_dataset(dataset, path=f'{SAVE_DIR}/fold_{i}/')
print(f'\nExperiment 2 splits generated and saved in {round((time.time()-start_time)/60, 2)} minutes.')

# Generate datasets featuring different levels of extrapolation, k folds each (experiment #3).
for conshift,extra in [(False, 'N'), (True, 'N'), (True,'E'), (True,'P')]:
    # Delete and remake save directory.
    CON_SHFT = conshift
    EXTRPLTE = extra
    SAVE_DIR = f'{SPLT_DIR}/A-{EXTRPLTE}-{HOLD_OUT}-{"S" if CON_SHFT else "N"}-{MAX_DIVR}'
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.mkdir(SAVE_DIR)
    # Generate and save k folds.
    for i in range(K_FOLDCV):
        # Increment random seeds per fold.
        random.seed(RAND_SED+i)
        numpy.random.seed(RAND_SED+i)
        # Re-collate and re-generate dataset per fold.
        image_data = collate_images()
        dataset    = generate_dataset(image_data)
        # Save fold to disk.
        save_dataset(dataset, path=f'{SAVE_DIR}/fold_{i}/')
print(f'\nExperiment 3 splits generated and saved in {round((time.time()-start_time)/60, 2)} minutes.')

print(f'\nFinished all tasks in {round((time.time()-start_time)/60, 2)} minutes.')
    
Set random seeds, collate images, and generate a new dataset.
random.seed(RAND_SED)
numpy.random.seed(RAND_SED)
start_time = time.time()
image_data = collate_images()
dataset    = generate_dataset(image_data)

Save dataset to disk.
save_dataset(dataset)
print(f'\nDataset generated and saved in {round((time.time()-start_time)/60, 2)} minutes.')

Load and display a random selection of problems from the new dataset.
observe_dataset()

Balance dataset for sampling in human survey.
balance_dataset()

Render dataset images to disk.
render_dataset()
print(f'\nFinished all tasks in {round((time.time()-start_time)/60, 2)} minutes.')


# END FILE ---------------------------------------------------------------------------------------------------------- #