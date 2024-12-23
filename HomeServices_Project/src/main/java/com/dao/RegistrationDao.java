package com.dao;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;

import com.model.UserModel;
import com.util.DBUtil;

public class RegistrationDao
{
	Connection cn=null;
	int x=0;
	public int userRegistration(UserModel rmodel)
	{
		cn=new DBUtil().getDatabaseConnection();
		String sql="insert into customer(firstname,lastname,gender,emailaddress,phonenumber,address,city,pincode,password) values(?,?,?,?,?,?,?,?,?)";
		
		try 
		{
			PreparedStatement st=cn.prepareStatement(sql);//pre compile the query
			st.setString(1, rmodel.getFirstname());
			st.setString(2, rmodel.getLastname());
			st.setString(3, rmodel.getGender());
			st.setString(4, rmodel.getEmailaddress());
			st.setString(5, rmodel.getPhonenumber());
			st.setString(6, rmodel.getAddress());
			st.setString(7, rmodel.getCity());
			st.setInt(8,(rmodel.getPincode()) );
			st.setString(9, rmodel.getPassword());
			//execute
			x=st.executeUpdate();
			cn.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return x;
	}
}