"""
Secure file upload endpoints.
"""
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import logging
from typing import Dict, Any

from ..middleware.auth_middleware import require_auth, get_current_user
from utils.security import secure_file_handler, audit_logger, InputValidator

logger = logging.getLogger(__name__)
upload_bp = Blueprint('upload', __name__)


@upload_bp.route('/upload/dataset', methods=['POST'])
@require_auth(['experiment:create'])
def upload_dataset() -> Dict[str, Any]:
    """
    Upload dataset file securely.
    
    Expected form data:
    - file: Dataset file (CSV, JSON, or JSONL)
    - name: Dataset name (optional)
    - description: Dataset description (optional)
    """
    try:
        current_user = get_current_user()
        user_id = current_user['user_id']
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'No file selected'
            }), 400
        
        # Get optional metadata
        dataset_name = request.form.get('name', '').strip()
        description = request.form.get('description', '').strip()
        
        # Validate metadata
        if dataset_name and not InputValidator.validate_filename(dataset_name):
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'Invalid dataset name'
            }), 400
        
        if description:
            description = InputValidator.sanitize_string(description, max_length=500)
        
        # Read file content
        file_content = file.read()
        original_filename = secure_filename(file.filename)
        
        # Save file securely
        file_info = secure_file_handler.save_uploaded_file(
            file_content=file_content,
            filename=original_filename,
            file_type='dataset',
            user_id=user_id
        )
        
        # Log successful upload
        audit_logger.log_data_access(
            user_id=user_id,
            resource_type='dataset',
            resource_id=file_info['secure_filename'],
            action='upload'
        )
        
        return jsonify({
            'success': True,
            'file_info': {
                'id': file_info['secure_filename'],
                'original_name': file_info['original_filename'],
                'size': file_info['size'],
                'mime_type': file_info['mime_type'],
                'uploaded_at': file_info['uploaded_at'],
                'name': dataset_name or original_filename,
                'description': description
            },
            'message': 'Dataset uploaded successfully'
        }), 201
        
    except ValueError as e:
        logger.warning(f"File upload validation failed: {e}")
        return jsonify({
            'success': False,
            'error': 'validation_error',
            'message': str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"File upload error: {e}")
        audit_logger.log_security_event(
            event_type="FILE_UPLOAD_ERROR",
            user_id=current_user.get('user_id') if 'current_user' in locals() else None,
            details=str(e),
            severity="ERROR"
        )
        return jsonify({
            'success': False,
            'error': 'upload_error',
            'message': 'Failed to upload file'
        }), 500


@upload_bp.route('/upload/model', methods=['POST'])
@require_auth(['model:create'])
def upload_model() -> Dict[str, Any]:
    """
    Upload model file securely.
    
    Expected form data:
    - file: Model file (ZIP or binary)
    - name: Model name
    - version: Model version (optional)
    - description: Model description (optional)
    """
    try:
        current_user = get_current_user()
        user_id = current_user['user_id']
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'No file selected'
            }), 400
        
        # Get required metadata
        model_name = request.form.get('name', '').strip()
        if not model_name:
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'Model name is required'
            }), 400
        
        # Validate model name
        if not InputValidator.validate_filename(model_name):
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'Invalid model name'
            }), 400
        
        # Get optional metadata
        version = request.form.get('version', '1.0.0').strip()
        description = request.form.get('description', '').strip()
        
        if description:
            description = InputValidator.sanitize_string(description, max_length=500)
        
        # Read file content
        file_content = file.read()
        original_filename = secure_filename(file.filename)
        
        # Save file securely
        file_info = secure_file_handler.save_uploaded_file(
            file_content=file_content,
            filename=original_filename,
            file_type='model',
            user_id=user_id
        )
        
        # Log successful upload
        audit_logger.log_data_access(
            user_id=user_id,
            resource_type='model',
            resource_id=file_info['secure_filename'],
            action='upload'
        )
        
        return jsonify({
            'success': True,
            'file_info': {
                'id': file_info['secure_filename'],
                'original_name': file_info['original_filename'],
                'size': file_info['size'],
                'mime_type': file_info['mime_type'],
                'uploaded_at': file_info['uploaded_at'],
                'name': model_name,
                'version': version,
                'description': description
            },
            'message': 'Model uploaded successfully'
        }), 201
        
    except ValueError as e:
        logger.warning(f"Model upload validation failed: {e}")
        return jsonify({
            'success': False,
            'error': 'validation_error',
            'message': str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Model upload error: {e}")
        audit_logger.log_security_event(
            event_type="MODEL_UPLOAD_ERROR",
            user_id=current_user.get('user_id') if 'current_user' in locals() else None,
            details=str(e),
            severity="ERROR"
        )
        return jsonify({
            'success': False,
            'error': 'upload_error',
            'message': 'Failed to upload model'
        }), 500


@upload_bp.route('/upload/<file_type>/<file_id>', methods=['DELETE'])
@require_auth(['model:delete', 'experiment:delete'])
def delete_uploaded_file(file_type: str, file_id: str) -> Dict[str, Any]:
    """
    Delete uploaded file securely.
    
    Args:
        file_type: Type of file (dataset, model, document)
        file_id: Secure file ID
    """
    try:
        current_user = get_current_user()
        user_id = current_user['user_id']
        
        # Validate file type
        if file_type not in ['dataset', 'model', 'document']:
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'Invalid file type'
            }), 400
        
        # Validate file ID
        if not InputValidator.validate_filename(file_id):
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'Invalid file ID'
            }), 400
        
        # Check if user owns the file (basic check - file ID contains user ID)
        if not file_id.startswith(user_id) and current_user.get('role') != 'admin':
            return jsonify({
                'success': False,
                'error': 'permission_denied',
                'message': 'You can only delete your own files'
            }), 403
        
        # Delete file
        success = secure_file_handler.delete_file(file_id, file_type, user_id)
        
        if success:
            audit_logger.log_data_access(
                user_id=user_id,
                resource_type=file_type,
                resource_id=file_id,
                action='delete'
            )
            
            return jsonify({
                'success': True,
                'message': 'File deleted successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'not_found',
                'message': 'File not found'
            }), 404
            
    except Exception as e:
        logger.error(f"File deletion error: {e}")
        return jsonify({
            'success': False,
            'error': 'deletion_error',
            'message': 'Failed to delete file'
        }), 500


@upload_bp.route('/upload/scan/<file_type>/<file_id>', methods=['POST'])
@require_auth(['system:monitor'])
def scan_uploaded_file(file_type: str, file_id: str) -> Dict[str, Any]:
    """
    Scan uploaded file for security threats.
    
    Args:
        file_type: Type of file (dataset, model, document)
        file_id: Secure file ID
    """
    try:
        current_user = get_current_user()
        user_id = current_user['user_id']
        
        # Validate inputs
        if file_type not in ['dataset', 'model', 'document']:
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'Invalid file type'
            }), 400
        
        if not InputValidator.validate_filename(file_id):
            return jsonify({
                'success': False,
                'error': 'validation_error',
                'message': 'Invalid file ID'
            }), 400
        
        # Get file path
        from pathlib import Path
        file_path = Path(secure_file_handler.upload_dir) / file_type / file_id
        
        if not file_path.exists():
            return jsonify({
                'success': False,
                'error': 'not_found',
                'message': 'File not found'
            }), 404
        
        # Scan file
        is_clean = secure_file_handler.scan_file_for_viruses(file_path)
        
        # Log scan result
        audit_logger.log_security_event(
            event_type="FILE_SCAN",
            user_id=user_id,
            details=f"file_id={file_id} clean={is_clean}",
            severity="INFO"
        )
        
        return jsonify({
            'success': True,
            'scan_result': {
                'file_id': file_id,
                'is_clean': is_clean,
                'scanned_at': audit_logger.logger.handlers[0].formatter.formatTime(
                    logging.LogRecord('', 0, '', 0, '', (), None)
                ) if audit_logger.logger.handlers else None
            },
            'message': 'File scan completed'
        }), 200
        
    except Exception as e:
        logger.error(f"File scan error: {e}")
        return jsonify({
            'success': False,
            'error': 'scan_error',
            'message': 'Failed to scan file'
        }), 500