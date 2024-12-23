package com.dao;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;

import com.model.UserModel;
import com.util.DBUtil;

public class HomeDao 
{
	Connection cn=null;
	public UserModel verifyLogin(int userid)
	{
		UserModel model=null;		
		cn=new DBUtil().getDatabaseConnection();
		String qry="select * from user where userid=?";
		try 
		{
			PreparedStatement st=cn.prepareStatement(qry);//pre-compiled the query 
			st.setInt(1, userid);
			ResultSet rs= st.executeQuery();
			if(rs.next())
			{
				model=new UserModel();
//				model.setUserid(rs.getInt(1));
				model.setFirstname(rs.getString("firstname"));
				model.setLastname(rs.getString(3));
//				model.setUsername(rs.getString(4));
//				model.setPassword(rs.getString(5));
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return model;
	}
}
