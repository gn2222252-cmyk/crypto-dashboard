# 加密貨幣儀表板：規格（主導端維護）

目的：一頁看完 BTC 為主、ETH 等為輔的市場狀態，並給出「規則目前的狀態」（偏多／中性／偏空）。
這是顯示工具，不是研究輪，也不是個人投資建議。

## 隔離規則（一定要守）

- 只放在 `dashboard/`。不經過 `src.data.loader.DataAccess`，不寫 `data/`，`src/research` 不得引用這裡。
- 只用免費、公開、不需金鑰的來源。User-Agent 固定 `btc-research-dashboard`（用 `common.get_json`）。不偽裝瀏覽器、不繞過機器人檢查；被擋就回報「失敗」，換別的公開來源。
- 不讀 `.env`，不用任何金鑰。
- **歷史只留前瞻期：** 寫進快取與頁面的 history 只留 2026-11-09 之後（`common.FORWARD_FROM`）；算均線用的長歷史只留在記憶體。顯示文字不得出現早於這天的事件日期。
- **不得當研究證據：** 快取、頁面與 `verdict.py` 的計票不得作為任何假設的來源，方向文件不得引用，不得據此開新家族。第二保留集（加密 2026-11-01 起）會隨前瞻期累積在快取裡，性質等同 live 紀錄，研究程式到 2027-11 前不得讀。
- **計票規則已固定**（每組一票、週期保險不計票、淨票數 ±2 才給方向），不得再改。頁面不出現買賣動作字眼。
- 執行時不要把歷史數列印到終端機（只印筆數與最新日期）。歷史只寫進 `dashboard/cache/`（不進版控）。
- 不要 commit、不要切分支、不要改 `dashboard/` 以外的檔案。只改自己負責的檔案。

## 收集器介面

每個收集器是 `dashboard/collectors/<name>.py`：

```python
from dashboard import common
def collect() -> list[dict]: ...          # 每個指標一個 common.indicator(...)；失敗用 common.failed(...)
if __name__ == "__main__":
    rows = collect(); common.save("<name>", rows)
    for r in rows: print(r["id"], r["asof"], r["display"] or r["error"])
```

- 執行：專案根目錄 `.venv/Scripts/python.exe -m dashboard.collectors.<name>`。
- **每個指標各自 try/except**，一個來源壞掉不能拖垮其他指標。
- 只用 `requests`、`pandas`、`numpy`（已安裝），不加新套件。
- 整個收集器要在 60 秒內跑完。

## 指標欄位（`common.indicator`）

| 欄位 | 說明 |
|---|---|
| id | 唯一，小寫底線，例如 `btc_funding_rate` |
| group | `price`／`derivatives`／`onchain`／`flows`／`sentiment` |
| asset | `BTC`、`ETH`、`SOL`、`MARKET`（整體市場） |
| name | 繁體中文短名稱，白話，例如「資金費率（幣安永續）」 |
| value / unit / display | 數值、單位、給人看的字串（例如 `0.012%／8 小時`） |
| asof | 這個數值的時間，UTC ISO 字串 |
| source / source_url | 來源名稱與人可以點開的網址 |
| evidence | `A` 本專案只在研發期過第一關、目前只做前瞻紀錄（只有週期保險的 PI、MVRV；不是通過的策略）；`B` 只有公開文獻；`C` 僅供參考。**不確定就給 C** |
| reading | `bullish`／`neutral`／`bearish`／`none`。只有在 `rule` 寫得出明確、事前固定的門檻時才給方向，否則 `none` |
| rule | 判讀規則的白話說明，含門檻數字與出處。門檻一律用慣例值或文獻值，**不得看資料調整** |
| note | 一句白話：這個指標在看什麼、限制是什麼 |
| history | `[[YYYY-MM-DD, 數值], ...]`，由舊到新，日頻，最多 400 筆 |

文字一律繁體中文、白話、不用縮寫（第一次出現要解釋）。
