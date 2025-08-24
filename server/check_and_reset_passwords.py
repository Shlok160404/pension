#!/usr/bin/env python3
"""
Script to check existing user passwords and reset them to known values for testing.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the Python path so we can import from app
current_dir = Path(__file__).parent
app_dir = current_dir / "app"
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(app_dir))

try:
    from app.database import SessionLocal
    from app.models import User
    from app.security import hash_password, verify_password
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

def check_and_reset_passwords():
    """Check existing passwords and reset them to known values."""
    
    db = SessionLocal()
    try:
        print("🔍 Checking existing user passwords...")
        
        # Get all users
        users = db.query(User).all()
        
        print(f"\n📊 Found {len(users)} users:")
        print("=" * 60)
        
        test_password = "password-123"
        updated_count = 0
        
        for user in users:
            print(f"\n👤 User ID: {user.id}")
            print(f"   📧 Email: {user.email}")
            print(f"   👤 Name: {user.full_name}")
            print(f"   🏷️  Role: {user.role}")
            print(f"   🔑 Password hash: {user.password[:50]}...")
            
            # Test if the test password works
            try:
                if verify_password(test_password, user.password):
                    print(f"   ✅ Password '{test_password}' works!")
                else:
                    print(f"   ❌ Password '{test_password}' does NOT work")
                    print(f"   🔄 Resetting password to '{test_password}'...")
                    
                    # Reset password
                    new_hash = hash_password(test_password)
                    user.password = new_hash
                    db.commit()
                    updated_count += 1
                    print(f"   ✅ Password reset successfully!")
                    
            except Exception as e:
                print(f"   ⚠️  Error verifying password: {e}")
                print(f"   🔄 Resetting password to '{test_password}'...")
                
                # Reset password due to hash error
                new_hash = hash_password(test_password)
                user.password = new_hash
                db.commit()
                updated_count += 1
                print(f"   ✅ Password reset successfully!")
        
        print(f"\n🎉 Password check/reset completed!")
        print(f"📊 Summary:")
        print(f"   Total users: {len(users)}")
        print(f"   Passwords reset: {updated_count}")
        print(f"\n🔑 All users now have password: {test_password}")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("🚀 Starting password check and reset script...")
    check_and_reset_passwords()
    print("\n✨ Script completed!")
