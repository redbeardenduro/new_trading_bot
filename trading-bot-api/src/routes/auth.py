from flask import Blueprint, jsonify, request
from flask_jwt_extended import (create_access_token, get_jwt_identity,
                                jwt_required)

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/api/auth/token", methods=["POST"])
def login():
    api_key = request.json.get("api_key", None)
    from production_trading_bot.core.config import BotConfig

    config = BotConfig()
    bootstrap_api_key = config.security.dash_api_key

    if not api_key or api_key != bootstrap_api_key:
        return jsonify({"msg": "Bad API key"}), 401

    access_token = create_access_token(identity="bootstrap_user")
    return jsonify(access_token=access_token)


# Example of a protected route
@auth_bp.route("/api/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200
