"""
JWT Authentication Module for Trading Bot API

Provides JWT-based authentication with role-based access control.
Supports 'viewer' and 'admin' roles for granular permission management.
"""

import os
import secrets
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, Optional

import jwt
from flask import jsonify, request


class AuthConfig:
    """Authentication configuration."""

    def __init__(self) -> None:
        self.jwt_secret = os.getenv("JWT_SECRET_KEY")
        if not self.jwt_secret:
            self.jwt_secret = secrets.token_urlsafe(32)
            print(
                "WARNING: JWT_SECRET_KEY not set in environment. Using random key (not suitable for production)."
            )
        self.jwt_algorithm = "HS256"
        self.token_expiry_hours = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
        self.api_key = os.getenv("API_KEY", "change-me-in-production")


auth_config = AuthConfig()


class Role:
    """Role definitions."""

    VIEWER = "viewer"
    ADMIN = "admin"
    ALL_ROLES = [VIEWER, ADMIN]


def generate_token(
    user_id: str, role: str = Role.VIEWER, custom_claims: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a JWT token for a user.

    Args:
        user_id: Unique identifier for the user
        role: User role (viewer or admin)
        custom_claims: Additional claims to include in the token

    Returns:
        JWT token string
    """
    if role not in Role.ALL_ROLES:
        raise ValueError(f"Invalid role: {role}. Must be one of {Role.ALL_ROLES}")
    now = datetime.utcnow()
    expiry = now + timedelta(hours=auth_config.token_expiry_hours)
    payload = {"user_id": user_id, "role": role, "iat": now, "exp": expiry, "nbf": now}
    if custom_claims:
        payload.update(custom_claims)
    token = jwt.encode(payload, auth_config.jwt_secret, algorithm=auth_config.jwt_algorithm)
    return token


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode a JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded token payload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, auth_config.jwt_secret, algorithms=[auth_config.jwt_algorithm])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_token_from_request() -> Optional[str]:
    """
    Extract JWT token from request headers.

    Supports both 'Authorization: Bearer <token>' and 'X-API-Token: <token>' headers.

    Returns:
        Token string if found, None otherwise
    """
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]
    token_header = request.headers.get("X-API-Token")
    if token_header:
        return token_header
    return None


def require_auth(required_role: Optional[str] = None) -> None:
    """
    Decorator to require authentication for a route.

    Args:
        required_role: Minimum role required (None = any authenticated user, 'admin' = admin only)

    Usage:
        @app.route('/admin/endpoint')
        @require_auth(required_role=Role.ADMIN)
        def admin_endpoint():
            return jsonify({'message': 'Admin only'})  # type: ignore[unreachable]
    """

    def decorator(f: Any) -> Any:

        @wraps(f)
        def decorated_function(*args: Any, **kwargs) -> Any:
            token = get_token_from_request()
            if not token:
                return (
                    jsonify({"error": "Authentication required", "message": "No token provided"}),
                    401,
                )
            payload = verify_token(token)
            if not payload:
                return (
                    jsonify(
                        {"error": "Authentication failed", "message": "Invalid or expired token"}
                    ),
                    401,
                )
            if required_role:
                user_role = payload.get("role")
                if user_role == Role.ADMIN:
                    pass
                elif required_role == Role.ADMIN and user_role != Role.ADMIN:
                    return (jsonify({"error": "Forbidden", "message": "Admin role required"}), 403)
            request.user_id = payload.get("user_id")
            request.user_role = payload.get("role")
            request.token_payload = payload
            return f(*args, **kwargs)

        return decorated_function

    return decorator  # type: ignore[return-value]


def verify_api_key(api_key: str) -> bool:
    """
    Verify an API key for initial token generation.

    Args:
        api_key: API key to verify

    Returns:
        True if valid, False otherwise
    """
    return api_key == auth_config.api_key


def create_auth_routes(app) -> None:
    """
    Create authentication routes for the Flask app.

    Args:
        app: Flask application instance
    """
    from flask import Blueprint

    auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

    @auth_bp.route("/token", methods=["POST"])
    def get_token() -> None:
        """
        Issue a JWT token in exchange for a valid API key.

        Request body:
        {
            "api_key": "your-api-key",
            "user_id": "optional-user-id",
            "role": "viewer|admin"
        }

        Response:
        {
            "token": "jwt-token-string",
            "expires_in": 86400,
            "role": "viewer"
        }
        """
        data = request.get_json()
        if not data:
            return (jsonify({"error": "Invalid request", "message": "JSON body required"}), 400)  # type: ignore[return-value]
        api_key = data.get("api_key")
        if not api_key:
            return (jsonify({"error": "Invalid request", "message": "api_key required"}), 400)  # type: ignore[return-value]
        if not verify_api_key(api_key):
            return (jsonify({"error": "Authentication failed", "message": "Invalid API key"}), 401)  # type: ignore[return-value]
        user_id = data.get("user_id", "default-user")
        role = data.get("role", Role.VIEWER)
        if role not in Role.ALL_ROLES:
            return (  # type: ignore[return-value]
                jsonify(
                    {
                        "error": "Invalid request",
                        "message": f"Invalid role. Must be one of {Role.ALL_ROLES}",
                    }
                ),
                400,
            )
        token = generate_token(user_id, role)
        return (  # type: ignore[return-value]
            jsonify(
                {
                    "token": token,
                    "expires_in": auth_config.token_expiry_hours * 3600,
                    "role": role,
                    "user_id": user_id,
                }
            ),
            200,
        )

    @auth_bp.route("/verify", methods=["GET"])
    @require_auth()
    def verify() -> None:
        """
        Verify the current token and return user information.

        Requires: Valid JWT token in Authorization header

        Response:
        {
            "user_id": "user-id",
            "role": "viewer",
            "valid": true
        }
        """
        return (  # type: ignore[return-value]
            jsonify({"valid": True, "user_id": request.user_id, "role": request.user_role}),
            200,
        )

    app.register_blueprint(auth_bp)
    return auth_bp
