# -*- coding: utf-8 -*-
"""
授權檢查：與主程式 app_for_offline.py 同一套「MAC + HMAC」規則，
抽成獨立模組，讓輕量版 server 不必為了驗證授權去 import 整包 torch。

優先序（與主程式一致）：實體乙太網卡 → Wi-Fi 永久位址 → 無固定 MAC。
Linux（Jetson）直接讀 /sys/class/net，不依賴 psutil；Windows 保留 psutil 後備。
"""
import hashlib
import hmac
import os
import sys

_SECRET = b"3DI-Pr0-X9k7-mZ2v-LwQn8-$ecret-2026"
_INVALID_MACS = {"", "00:00:00:00:00:00", "FF:FF:FF:FF:FF:FF"}
_NAME_HINTS_SKIP = ("lo", "docker", "veth", "br-", "virbr", "l4tbr", "tun", "tap", "vmnet", "can")

_cache = None


def _normalize_mac(mac: str) -> str:
    mac = (mac or "").strip().replace("-", ":").upper()
    return mac if len(mac) == 17 else ""


def _is_universal(mac: str) -> bool:
    """本機自訂（locally administered）位址 → 不可用來綁授權（docker / usb gadget / 隨機 MAC）。"""
    try:
        return (int(mac.split(":")[0], 16) & 0x02) == 0
    except (ValueError, IndexError):
        return False


def _linux_macs():
    """回傳 (ethernet_macs, wifi_macs)：只取有實體 device 的介面，並排除虛擬/本機自訂位址。"""
    eth, wifi = [], []
    net_dir = "/sys/class/net"
    if not os.path.isdir(net_dir):
        return eth, wifi
    for name in sorted(os.listdir(net_dir)):
        if any(name.startswith(h) for h in _NAME_HINTS_SKIP):
            continue
        base = os.path.join(net_dir, name)
        if not os.path.exists(os.path.join(base, "device")):
            continue                                   # 無實體裝置 → 虛擬介面
        try:
            with open(os.path.join(base, "address")) as f:
                mac = _normalize_mac(f.read())
            with open(os.path.join(base, "type")) as f:
                iftype = f.read().strip()
        except OSError:
            continue
        if not mac or mac in _INVALID_MACS or not _is_universal(mac):
            continue
        if os.path.isdir(os.path.join(base, "wireless")) or name.startswith("wl"):
            wifi.append(mac)
        elif iftype == "1":                            # ARPHRD_ETHER
            eth.append(mac)
    return eth, wifi


def _psutil_macs():
    """非 Linux 後備：以介面名稱排除無線/虛擬，只留原廠燒錄的實體 MAC。"""
    eth = []
    try:
        import psutil
        skip = ("wi-fi", "wifi", "wireless", "802.11", "wlan", "bluetooth", "loopback",
                "virtual", "vmware", "virtualbox", "vbox", "hyper-v", "tap", "tun", "vpn", "docker")
        for name, addrs in psutil.net_if_addrs().items():
            if any(h in str(name).lower() for h in skip):
                continue
            for a in addrs:
                if getattr(a, "family", None) == getattr(psutil, "AF_LINK", None):
                    m = _normalize_mac(a.address or "")
                    if m and m not in _INVALID_MACS and _is_universal(m):
                        eth.append(m)
    except Exception:
        pass
    return eth, []


def get_license_macs():
    """(macs, source)；source ∈ {'ethernet', 'wifi', 'none'}。結果快取一次。"""
    global _cache
    if _cache is not None:
        return _cache
    eth, wifi = _linux_macs() if sys.platform.startswith("linux") else _psutil_macs()
    if eth:
        _cache = (sorted(set(eth)), "ethernet")
    elif wifi:
        _cache = (sorted(set(wifi)), "wifi")
    else:
        _cache = ([], "none")
    return _cache


def machine_mac() -> str:
    macs, _ = get_license_macs()
    return macs[0] if macs else ""


def check_license(search_dirs) -> tuple:
    """(licensed:bool, message:str, mac:str)。license.txt 可放在任一 search_dir。"""
    path = None
    for d in search_dirs:
        for cand in (os.path.join(d, "license.txt"), os.path.join(d, "settings", "license.txt")):
            if os.path.isfile(cand):
                path = cand
                break
        if path:
            break
    if path is None:
        return False, "找不到 license.txt，請向 AML 申請授權檔案", machine_mac()
    try:
        data = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                data[k.strip().upper()] = v.strip()
        mac_in_file = _normalize_mac(data.get("MACHINE", ""))
        key_in_file = data.get("KEY", "")
        current, _src = get_license_macs()
        if not mac_in_file or not key_in_file:
            return False, "license.txt 格式錯誤", machine_mac()
        if not current:
            return False, "本機沒有固定 MAC（無實體網卡），無法綁定授權", ""
        if mac_in_file not in current:
            return False, f"授權碼不屬於本機（本機 MAC：{machine_mac()}）", machine_mac()
        expected = hmac.new(_SECRET, mac_in_file.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(key_in_file, expected):
            return False, "授權碼無效，請確認 license.txt 內容", machine_mac()
        return True, "授權有效", machine_mac()
    except Exception as e:
        return False, f"授權檔案讀取失敗：{e}", machine_mac()
