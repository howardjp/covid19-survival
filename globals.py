import os
import spdlog

global logger

log_level_map = {"trace": spdlog.LogLevel.TRACE,
                 "debug": spdlog.LogLevel.DEBUG,
                 "info": spdlog.LogLevel.INFO,
                 "warn": spdlog.LogLevel.WARN,
                 "err": spdlog.LogLevel.ERR,
                 "critical": spdlog.LogLevel.CRITICAL,
                 "off": spdlog.LogLevel.OFF}

info_banner = "James Howard's ACM Thesis Project"
app_name = "c19cox"

logger = spdlog.ConsoleLogger(app_name)