package com.dao;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import com.model.AssignModel;
import com.model.BookModel;
import com.model.EmployeeModel;
import com.model.FeedbackModel;
import com.model.UserModel;
import com.util.DBUtil;

public class LoginDao 
{
	Connection cn=null;
	public UserModel getLogin(UserModel lmodel)
	{
		UserModel model=null;		
		cn=new DBUtil().getDatabaseConnection();
		String qry="select * from customer where emailaddress=? and password=?";
		try 
		{
			PreparedStatement st=cn.prepareStatement(qry);//pre compiled the query 
			st.setString(1, lmodel.getEmailaddress());
			st.setString(2, lmodel.getPassword());
			ResultSet rs= st.executeQuery();
			if(rs.next())
			{
				model=new UserModel();
				model.setCustomerid(rs.getInt(1));
				model.setFirstname(rs.getString("firstname"));
				model.setLastname(rs.getString(3));
				model.setGender(rs.getString(4));
				model.setEmailaddress(rs.getString(5));
				model.setPhonenumber(rs.getString(6));
				model.setAddress(rs.getString(7));
				model.setCity(rs.getString(8));
				model.setPincode(rs.getInt(9));
				
				model.setPassword(rs.getString(10));
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return model;
	}
	
	public String getTitle(int id)
	{
			String title="";	
		cn=new DBUtil().getDatabaseConnection();
		String qry="select title from subservicei where id=?";
		try 
		{
			PreparedStatement st=cn.prepareStatement(qry);//pre compiled the query 
			st.setInt(1, id);
			ResultSet rs= st.executeQuery();
			if(rs.next())
			{
			title = rs.getString(1); 	
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return title;
	}
	
	
	public int customerBooking(BookModel bmodel)
	{
		int x=0;
		cn=new DBUtil().getDatabaseConnection();
		String sql="insert into book(id,userid,bdate,price,status,pstatus) values(?,?,?,?,?,?)";
		
		try 
		{
			PreparedStatement st=cn.prepareStatement(sql);//pre compile the query
			st.setInt(1, bmodel.getId());
			st.setInt(2, bmodel.getUserid());
			st.setString(3, bmodel.getBdate());
			st.setString(4, bmodel.getPrice());
			st.setString(5, bmodel.getStatus());
			st.setString(6, bmodel.getPstatus());

			//execute
			x=st.executeUpdate();
			cn.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return x;
	}
	public EmployeeModel getEmployeeLogin(EmployeeModel ep)
	{
		EmployeeModel model=null;		
		cn=new DBUtil().getDatabaseConnection();
		String qry="select * from employee where email=? and password=?";
		try 
		{
			PreparedStatement st=cn.prepareStatement(qry);//pre compiled the query 
			st.setString(1, ep.getEmail());
			st.setString(2, ep.getPassword());
			ResultSet rs= st.executeQuery();
			if(rs.next())
			{
				model=new EmployeeModel();
				model.setFirstname(rs.getString("firstname"));
				model.setLastname(rs.getString("lastname"));
				model.setEmail(rs.getString("email"));
				model.setPhone(rs.getString("phone"));
				model.setPassword(rs.getString("password"));
				model.setSpecialization(rs.getString("specialization"));
				model.setGender(rs.getString("gender"));
				}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return model;
	}
	public int feedback(FeedbackModel fmodel)
	{
		int x=0;
		cn=new DBUtil().getDatabaseConnection();
		String sql="insert into feedback(userid,fullname,subject,email,comment) values(?,?,?,?,?)";
		
		try 
		{
			PreparedStatement st=cn.prepareStatement(sql);//pre compile the query
			st.setInt(1, fmodel.getUserid());
			st.setString(2, fmodel.getFullname());
			st.setString(3, fmodel.getSubject());
			st.setString(4, fmodel.getEmail());

			st.setString(5, fmodel.getComment());

			//execute
			x=st.executeUpdate();
			cn.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return x;
	}
	public int feedback1(FeedbackModel fmodel)
	{
		int x=0;
		cn=new DBUtil().getDatabaseConnection();
		String sql="insert into feedback(fullname,subject,email,comment) values(?,?,?,?)";
		
		try 
		{
			PreparedStatement st=cn.prepareStatement(sql);//pre compile the query
//			st.setInt(1, fmodel.getUserid());
			st.setString(1, fmodel.getFullname());
			st.setString(2, fmodel.getSubject());
			st.setString(3, fmodel.getEmail());

			st.setString(4, fmodel.getComment());

			//execute
			x=st.executeUpdate();
			cn.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return x;
	}
	public int update_profile(UserModel fmodel)
	{
		int x=0;
		cn=new DBUtil().getDatabaseConnection();
		String sql="update customer set firstname=?,lastname=?,emailaddress=?,phonenumber=?,address=?,pincode=? where userid=?";
		
		try 
		{
			PreparedStatement st=cn.prepareStatement(sql);//pre compile the query
			st.setString(1, fmodel.getFirstname());
			st.setString(2, fmodel.getLastname());
			st.setString(3, fmodel.getEmailaddress());
			st.setString(4, fmodel.getPhonenumber());

			st.setString(5, fmodel.getAddress());
			st.setInt(6, fmodel.getPincode());
			st.setInt(7, fmodel.getCustomerid());
			

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
