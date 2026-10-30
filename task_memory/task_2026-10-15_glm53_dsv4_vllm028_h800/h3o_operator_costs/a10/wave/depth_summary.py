import ast, sys, statistics
d = [ast.literal_eval(l)['depth'] for l in open('depth_out.txt') if l.startswith('{')]
inr = [x for x in d if 18.0 <= x <= 19.0]
print('waves', len(d), 'in [18,19]', len(inr), 'min/max in range', min(inr), max(inr), 'median', statistics.median(d))
print('outside', sorted(x for x in d if not 18.0 <= x <= 19.0))
