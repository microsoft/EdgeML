# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os
from importlib import import_module

IS_QVGA_MONO = os.environ['IS_QVGA_MONO']


name = 'config'
if IS_QVGA_MONO == '1':
	name = name + '_qvga'


cfg = import_module('data.' + name)
