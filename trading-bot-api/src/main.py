import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env file in development
if os.environ.get("FLASK_ENV") == "development":
    load_dotenv()
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, Response, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required
from production_trading_bot.core.config import BotConfig
from src.database import db
from src.routes.advanced_charting import advanced_charting_bp
from src.routes.auth import auth_bp
from src.routes.backtesting import BacktestNamespace, backtesting_bp
from src.routes.console import console_bp, init_socketio, simulate_bot_logs
from src.routes.enhanced_backtesting import enhanced_backtesting_bp
from src.routes.market_intelligence import market_intelligence_bp
from src.routes.portfolio_analytics import portfolio_analytics_bp
from src.routes.trading_bot import trading_bot_bp
from src.routes.user import user_bp

# Import Prometheus metrics
try:
    # Adjust path to reach core directory
    core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if core_path not in sys.path:
        sys.path.insert(0, core_path)
    from production_trading_bot.core.metrics import (get_content_type,
                                                     get_metrics)

    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Prometheus metrics not available: {e}")
    METRICS_AVAILABLE = False


app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"))
app.config["SECRET_KEY"] = "asdf#FGSgvasgf$5$WGT"
config = BotConfig()
app.config["JWT_SECRET_KEY"] = (
    config.security.api_jwt_secret or "super-secret-jwt-key"
)  # Fallback for dev key
jwt = JWTManager(app)


# Enable CORS for all routes
CORS(app)

# Initialize SocketIO
socketio = init_socketio(app)

app.register_blueprint(user_bp, url_prefix="/api")
app.register_blueprint(trading_bot_bp, url_prefix="/api/bot")
app.register_blueprint(console_bp, url_prefix="/api/console")
app.register_blueprint(backtesting_bp)
app.register_blueprint(portfolio_analytics_bp)
app.register_blueprint(enhanced_backtesting_bp)
app.register_blueprint(market_intelligence_bp)
app.register_blueprint(advanced_charting_bp)
app.register_blueprint(auth_bp, url_prefix="/api/auth")

# Register backtest namespace
socketio.on_namespace(BacktestNamespace("/backtest"))

# Register enhanced features


# uncomment if you need to use database
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'app.db')}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)
with app.app_context():
    db.create_all()

from production_trading_bot.core.metrics import track_internal_api_call


@app.route("/metrics")
@jwt_required()
@track_internal_api_call(endpoint="/metrics", method="GET")
def metrics():
    """Prometheus metrics endpoint (protected)."""
    if not METRICS_AVAILABLE:
        return Response("Metrics not available", status=503)

    return Response(get_metrics(), mimetype=get_content_type())


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
        return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, "index.html")
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, "index.html")
        else:
            return "index.html not found", 404


if __name__ == "__main__":
    # Start the log simulator for demonstration
    simulate_bot_logs()

    # Run with SocketIO
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
