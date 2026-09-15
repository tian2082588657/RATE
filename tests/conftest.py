# -*- coding: utf-8 -*-
"""conftest.py — 让 tests/ 能 import 仓库根目录模块。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
