"""Paper trading execution via Alpaca API."""

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import get_settings
import logging

logger = logging.getLogger(__name__)


class PaperTrader:
    """Execute paper trades through Alpaca."""

    def __init__(self):
        settings = get_settings()
        self.client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=True,
        )

    def get_account(self) -> dict:
        """Get account info (cash, equity, P&L)."""
        account = self.client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "pnl": float(account.equity) - float(account.last_equity),
        }

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        positions = self.client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pnl": float(p.unrealized_pl),
                "pnl_pct": float(p.unrealized_plpc),
            }
            for p in positions
        ]

    def submit_order(
        self, symbol: str, qty: float, side: str = "buy"
    ) -> dict:
        """Submit a market order."""
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        order = self.client.submit_order(order_data)
        logger.info(f"Order submitted: {side} {qty} {symbol} | ID: {order.id}")
        return {
            "id": str(order.id),
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "status": str(order.status),
        }

    def close_position(self, symbol: str) -> dict:
        """Close an entire position."""
        self.client.close_position(symbol)
        logger.info(f"Position closed: {symbol}")
        return {"symbol": symbol, "status": "closed"}

    def close_all(self) -> dict:
        """Close all positions."""
        self.client.close_all_positions(cancel_orders=True)
        logger.info("All positions closed")
        return {"status": "all_closed"}
