package com.dao;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;

import com.model.UserModel;
import com.util.DBUtil;

public class ForgotDao {
	
	public UserModel checkEmail(String emailaddress)
	{
		UserModel f=new UserModel();
		try {
		
			Connection conn=new DBUtil().getDatabaseConnection();
			String sql="select * from  customer where emailaddress=?";
			PreparedStatement stmt=conn.prepareStatement(sql);
			stmt.setString(1, emailaddress);
			ResultSet rs=stmt.executeQuery();
			if(rs.next())
			{
				f.setCustomerid(rs.getInt(1));
				f.setEmailaddress(emailaddress);
				f.setFirstname(rs.getString(2));
				f.setLastname(rs.getString(3));
			}
			else

			{
				f=null;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return f;
		
	}
	public int resetPassword(UserModel f)
	{
		int r=0;
		try {
			Connection conn=new DBUtil().getDatabaseConnection();
			String sql="update customer set password=? where userid=?";
			PreparedStatement stmt=conn.prepareStatement(sql);
			stmt.setString(1, f.getPassword());
			stmt.setInt(2, f.getCustomerid());
			r=stmt.executeUpdate();
			conn.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return r;
	}

}