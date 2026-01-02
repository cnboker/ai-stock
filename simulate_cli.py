# simulate_cli.py
import argparse
from simulator.replay_trade_day import simulate_trade_day

#python3 simulate_cli.py --ticker=sh600446 --date=2025-12-31
def main():
    parser = argparse.ArgumentParser(description="Simulate a trading day and compare with real trades")
    parser.add_argument("--ticker", required=True, help="Ticker symbol")
    parser.add_argument("--date", required=True, help="Date to simulate YYYY-MM-DD")
    args = parser.parse_args()

    snapshots = simulate_trade_day(args.ticker,args.date)
  
if __name__ == "__main__":
    main()
