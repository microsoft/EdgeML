import os
from importlib import import_module

IS_QVGA_MONO = os.environ['IS_QVGA_MONO'] if 'IS_QVGA_MONO' in os.environ else '0'

name = 'config'
if IS_QVGA_MONO == '1':
	name = name + '_qvga'


cfg = import_module('data.' + name)
